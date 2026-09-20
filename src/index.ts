/*
 * Minimal GraphQL execution core focused on correct non-null propagation.
 *
 * The completion walk mirrors the GraphQL "Complete Value" algorithm:
 *  1. When a field fails (resolver throws, or a non-null wrapper sees null),
 *     an error is recorded *at that field's immutable path* first.
 *  2. A non-null violation then bubbles outward one wrapper at a time:
 *       - nullable list item slot  -> that element becomes null
 *       - nullable list           -> the whole list becomes null
 *       - nullable object field   -> that field becomes null
 *       - non-null wrapper        -> keep bubbling
 *     If it reaches the root, `data` becomes null.
 *  3. Sibling fields / list elements run in parallel. The first violation
 *     that makes a scope unusable aborts that scope's AbortSignal (and only
 *     that scope), so cooperative sibling tasks may stop; errors already
 *     collected are kept and returned in stable query-field order.
 */

// ---------------------------------------------------------------------------
// Type system
// ---------------------------------------------------------------------------

export type TypeRef = {
  kind: 'named' | 'list' | 'nonNull';
  name?: string;
  ofType?: TypeRef;
};

export interface GraphQLResolveInfo {
  fieldName: string;
  responseKey: string;
  path: (string | number)[];
  parentTypeName: string;
  signal: AbortSignal | undefined;
}

export interface GraphQLField {
  type: TypeRef;
  resolve?: (
    source: unknown,
    context: unknown,
    info: GraphQLResolveInfo,
  ) => unknown | Promise<unknown>;
}

export interface GraphQLObjectType {
  kind: 'object';
  name: string;
  fields: Record<string, GraphQLField>;
}

export interface GraphQLScalarType {
  kind: 'scalar';
  name: string;
}

export type GraphQLNamedType = GraphQLObjectType | GraphQLScalarType;

export interface GraphQLSchema {
  query: string;
  types: Record<string, GraphQLNamedType>;
}

export const named = (name: string): TypeRef => ({ kind: 'named', name });
export const list = (ofType: TypeRef): TypeRef => ({ kind: 'list', ofType });
export const nonNull = (ofType: TypeRef): TypeRef => ({ kind: 'nonNull', ofType });

// ---------------------------------------------------------------------------
// Query document (tiny hand-written AST; aliases + nested selection sets)
// ---------------------------------------------------------------------------

export interface FieldNode {
  kind?: 'field';
  name: string;
  alias?: string;
  selectionSet?: FieldNode[];
}

export interface DocumentNode {
  kind?: 'document';
  selectionSet: FieldNode[];
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

export interface GraphQLErrorOptions {
  path?: (string | number)[];
  extensions?: Record<string, unknown>;
}

export class GraphQLError extends Error {
  path: (string | number)[];
  readonly extensions: Record<string, unknown>;
  /** Ordinal key (query-field order interleaved with list indices). */
  orderKey: number[] = [];
  /** Insertion sequence, used as a stable tie-breaker. */
  seq = 0;

  constructor(message: string, options: GraphQLErrorOptions = {}) {
    super(message);
    this.name = 'GraphQLError';
    this.path = options.path ? [...options.path] : [];
    this.extensions = options.extensions ? { ...options.extensions } : {};
  }
}

export interface ExecutionResult {
  data: Record<string, unknown> | null;
  errors: GraphQLError[];
}

// ---------------------------------------------------------------------------
// Immutable response path
// ---------------------------------------------------------------------------

class Path {
  constructor(
    readonly prev: Path | null,
    /** Response key: field alias/name, or list index. */
    readonly key: string | number,
    /**
     * Ordering ordinal: position of the field node in its selection set,
     * or the list index itself.
     */
    readonly ord: number,
  ) {}
}

function pathToArray(path: Path | null): (string | number)[] {
  const out: (string | number)[] = [];
  let cur: Path | null = path;
  while (cur) {
    out.push(cur.key);
    cur = cur.prev;
  }
  return out.reverse();
}

function pathToOrderKey(path: Path | null): number[] {
  const out: number[] = [];
  let cur: Path | null = path;
  while (cur) {
    out.push(cur.ord);
    cur = cur.prev;
  }
  return out.reverse();
}

// ---------------------------------------------------------------------------
// Internal markers
// ---------------------------------------------------------------------------

/** Thrown when a non-null wrapper cannot be satisfied; must bubble. */
const NON_NULL_VIOLATION: unique symbol = Symbol('non-null violation');
/** A unit that stopped cooperatively because its scope was aborted. */
const CANCELLED: unique symbol = Symbol('cancelled');

type Violation = typeof NON_NULL_VIOLATION;
type Cancelled = typeof CANCELLED;

interface ExecutionContext {
  schema: GraphQLSchema;
  contextValue: unknown;
  errors: GraphQLError[];
  nextSeq: number;
}

interface UnitResult {
  value: unknown;
}

function isCancellation(error: unknown, signal: AbortSignal | undefined): boolean {
  if (error === CANCELLED) return true;
  const e = error as { name?: string; code?: string } | null;
  return (
    !!signal?.aborted &&
    (e?.name === 'AbortError' || e?.code === 'ABORT_ERR')
  );
}

// ---------------------------------------------------------------------------
// Error reporting: location is fixed at the originally failing field and
// never rewritten while the violation bubbles.
// ---------------------------------------------------------------------------

function reportThrown(
  ctx: ExecutionContext,
  thrown: unknown,
  path: Path | null,
): void {
  let error: GraphQLError;
  if (thrown instanceof GraphQLError) {
    error = new GraphQLError(thrown.message, {
      path: thrown.path.length > 0 ? thrown.path : pathToArray(path),
      extensions: thrown.extensions,
    });
  } else {
    const e = thrown as
      | { message?: string; extensions?: Record<string, unknown> }
      | null
      | undefined;
    error = new GraphQLError(e?.message ?? String(thrown), {
      path: pathToArray(path),
      extensions: e?.extensions,
    });
  }
  error.orderKey = pathToOrderKey(path);
  error.seq = ctx.nextSeq++;
  ctx.errors.push(error);
}

function reportNonNullNull(ctx: ExecutionContext, path: Path | null): void {
  const segments = pathToArray(path);
  const where = segments.length > 0 ? ` at path "${segments.join('.')}"` : '';
  const error = new GraphQLError(
    `Cannot return null for non-nullable value${where}.`,
    { path: segments },
  );
  error.orderKey = pathToOrderKey(path);
  error.seq = ctx.nextSeq++;
  ctx.errors.push(error);
}

// ---------------------------------------------------------------------------
// Scoped cancellation
// ---------------------------------------------------------------------------

function childController(parent: AbortSignal | undefined): AbortController {
  const ac = new AbortController();
  if (parent) {
    if (parent.aborted) ac.abort();
    else
      parent.addEventListener('abort', () => ac.abort(), { once: true });
  }
  return ac;
}

/**
 * Run independent units (object fields or list elements) in parallel.
 * - A unit resolves to {value} when nullable-safe.
 * - A unit returns CANCELLED when it observed scope abort before producing.
 * - A unit rejects with NON_NULL_VIOLATION when a non-null boundary is hit.
 *   The first such violation aborts the scope; remaining in-flight units may
 *   finish cooperatively, but their post-abort results are discarded.
 * Errors themselves live in the shared ExecutionContext sink, so everything
 * already reported survives regardless of how propagation settles.
 */
async function runParallel(
  units: Array<(signal: AbortSignal) => Promise<UnitResult | Cancelled>>,
  parentSignal: AbortSignal | undefined,
): Promise<unknown[]> {
  const ac = childController(parentSignal);
  const n = units.length;
  const outcomes = new Array<UnitResult | Cancelled>(n);
  if (n === 0) return [];

  let pending = n;
  let dead = false;

  await new Promise<void>((resolve, reject) => {
    let settled = false;
    const done = () => {
      pending -= 1;
      if (pending > 0) return;
      if (settled) return;
      settled = true;
      if (dead) reject(NON_NULL_VIOLATION);
      else resolve();
    };

    units.forEach((unit, i) => {
      if (ac.signal.aborted) {
        outcomes[i] = CANCELLED;
        queueMicrotask(done);
        return;
      }
      unit(ac.signal).then(
        (result) => {
          outcomes[i] = result;
          done();
        },
        (error) => {
          if (error === NON_NULL_VIOLATION) {
            // This scope is now unusable: cancel sibling work.
            dead = true;
            ac.abort();
          }
          outcomes[i] = CANCELLED;
          done();
        },
      );
    });
  });

  return outcomes.map((outcome) =>
    outcome === CANCELLED ? null : (outcome as UnitResult).value,
  );
}

// ---------------------------------------------------------------------------
// Field execution (object-field nullability boundary)
// ---------------------------------------------------------------------------

function makeInfo(
  parentType: GraphQLObjectType,
  node: FieldNode,
  path: Path,
  signal: AbortSignal | undefined,
): GraphQLResolveInfo {
  return {
    fieldName: node.name,
    responseKey: node.alias ?? node.name,
    path: pathToArray(path),
    parentTypeName: parentType.name,
    signal,
  };
}

async function executeField(
  parentType: GraphQLObjectType,
  node: FieldNode,
  source: unknown,
  path: Path | null,
  ordinal: number,
  ctx: ExecutionContext,
  parentSignal: AbortSignal | undefined,
): Promise<UnitResult | Cancelled> {
  if (parentSignal?.aborted) return CANCELLED;

  const responseKey = node.alias ?? node.name;
  const fieldPath = new Path(path, responseKey, ordinal);
  const field = parentType.fields[node.name];

  if (!field) {
    reportThrown(
      ctx,
      new GraphQLError(
        `Cannot query field "${node.name}" on type "${parentType.name}".`,
      ),
      fieldPath,
    );
    return { value: null };
  }

  let raw: unknown;
  try {
    if (field.resolve) {
      raw = await field.resolve(source, ctx.contextValue, makeInfo(parentType, node, fieldPath, parentSignal));
    } else {
      raw =
        source == null
          ? null
          : (source as Record<string, unknown>)[node.name];
    }
  } catch (error) {
    if (error === NON_NULL_VIOLATION) throw error;
    if (isCancellation(error, parentSignal)) return CANCELLED;
    // 1. pin the error at this field first ...
    reportThrown(ctx, error, fieldPath);
    // 2. ... then decide propagation from this field's wrapper.
    if (field.type.kind === 'nonNull') throw NON_NULL_VIOLATION;
    return { value: null };
  }

  try {
    const value = await complete(
      field.type,
      raw,
      fieldPath,
      ctx,
      parentSignal,
      node.selectionSet,
    );
    return { value };
  } catch (error) {
    if (error === CANCELLED) return CANCELLED;
    if (error === NON_NULL_VIOLATION) {
      // Violation from below: non-null field keeps bubbling,
      // nullable field is the boundary that becomes null.
      if (field.type.kind === 'nonNull') throw NON_NULL_VIOLATION;
      return { value: null };
    }
    if (isCancellation(error, parentSignal)) return CANCELLED;
    reportThrown(ctx, error, fieldPath);
    if (field.type.kind === 'nonNull') throw NON_NULL_VIOLATION;
    return { value: null };
  }
}

// ---------------------------------------------------------------------------
// Object completion (parent-object nullability is decided by the caller)
// ---------------------------------------------------------------------------

async function executeObject(
  type: GraphQLObjectType,
  source: unknown,
  path: Path | null,
  ctx: ExecutionContext,
  signal: AbortSignal | undefined,
  nodes: FieldNode[],
): Promise<Record<string, unknown>> {
  const values = await runParallel(
    nodes.map((node, ordinal) => (innerSignal) =>
      executeField(type, node, source, path, ordinal, ctx, innerSignal),
    ),
    signal,
  );
  const result: Record<string, unknown> = {};
  nodes.forEach((node, i) => {
    result[node.alias ?? node.name] = values[i];
  });
  return result;
}

// ---------------------------------------------------------------------------
// List completion (element/list nullability boundary)
// ---------------------------------------------------------------------------

function coerceIterable(value: unknown): unknown[] {
  if (Array.isArray(value)) return value;
  if (typeof value !== 'string' && value != null && Symbol.iterator in Object(value)) {
    return Array.from(value as Iterable<unknown>);
  }
  // GraphQL coerces a non-list value into a single-element list.
  return [value];
}

async function completeList(
  itemType: TypeRef,
  listValue: unknown,
  path: Path | null,
  ctx: ExecutionContext,
  signal: AbortSignal | undefined,
  selection: FieldNode[] | undefined,
): Promise<unknown[]> {
  const items = coerceIterable(listValue);
  return runParallel(
    items.map((item, index) => async (innerSignal): Promise<UnitResult | Cancelled> => {
      // Each parallel element owns its own immutable path.
      const itemPath = new Path(path, index, index);
      try {
        const value = await complete(
          itemType,
          item,
          itemPath,
          ctx,
          innerSignal,
          selection,
        );
        return { value };
      } catch (error) {
        if (error === CANCELLED) return CANCELLED;
        if (error === NON_NULL_VIOLATION) {
          // Non-null element type: the element cannot be null, so the whole
          // list is dead and the violation bubbles to the list's caller.
          // Nullable element type: only this element becomes null.
          if (itemType.kind === 'nonNull') throw NON_NULL_VIOLATION;
          return { value: null };
        }
        if (isCancellation(error, innerSignal)) return CANCELLED;
        reportThrown(ctx, error, itemPath);
        if (itemType.kind === 'nonNull') throw NON_NULL_VIOLATION;
        return { value: null };
      }
    }),
    signal,
  );
}

// ---------------------------------------------------------------------------
// Complete value: recurse through the wrapper types one layer at a time
// ---------------------------------------------------------------------------

async function complete(
  type: TypeRef,
  value: unknown,
  path: Path | null,
  ctx: ExecutionContext,
  signal: AbortSignal | undefined,
  selection: FieldNode[] | undefined,
): Promise<unknown> {
  if (signal?.aborted) throw CANCELLED;

  if (type.kind === 'nonNull') {
    if (value == null) {
      reportNonNullNull(ctx, path);
      throw NON_NULL_VIOLATION;
    }
    const inner = await complete(type.ofType!, value, path, ctx, signal, selection);
    if (inner == null) {
      reportNonNullNull(ctx, path);
      throw NON_NULL_VIOLATION;
    }
    return inner;
  }

  // Wrapper from here on is nullable: null simply propagates as a value.
  if (value == null) return null;

  if (type.kind === 'list') {
    return completeList(type.ofType!, value, path, ctx, signal, selection);
  }

  const namedType = type.name ? ctx.schema.types[type.name] : undefined;
  if (namedType && namedType.kind === 'object') {
    return executeObject(namedType, value, path, ctx, signal, selection ?? []);
  }
  // Scalar / enum: pass through.
  return value;
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

export interface ExecuteOptions {
  rootValue?: unknown;
  contextValue?: unknown;
  signal?: AbortSignal;
}

export async function execute(
  schema: GraphQLSchema,
  document: DocumentNode,
  options: ExecuteOptions = {},
): Promise<ExecutionResult> {
  const queryType = schema.types[schema.query];
  if (!queryType || queryType.kind !== 'object') {
    throw new Error(`Schema is missing object query type "${schema.query}".`);
  }

  const ctx: ExecutionContext = {
    schema,
    contextValue: options.contextValue ?? {},
    errors: [],
    nextSeq: 0,
  };

  let data: Record<string, unknown> | null;
  try {
    data = await executeObject(
      queryType,
      options.rootValue ?? {},
      null,
      ctx,
      options.signal,
      document.selectionSet,
    );
  } catch (error) {
    if (error === NON_NULL_VIOLATION) {
      // Violation escaped every nullable wrapper: data itself is null.
      data = null;
    } else if (isCancellation(error, options.signal)) {
      data = null;
    } else {
      reportThrown(ctx, error, null);
      data = null;
    }
  }

  // Stable query-field order, interleaved with list indices; insertion order
  // breaks ties so two errors at the same path keep collection order.
  const errors = [...ctx.errors].sort((a, b) => {
    const len = Math.max(a.orderKey.length, b.orderKey.length);
    for (let i = 0; i < len; i++) {
      const x = a.orderKey[i] ?? -1;
      const y = b.orderKey[i] ?? -1;
      if (x !== y) return x - y;
    }
    return a.seq - b.seq;
  });

  return { data, errors };
}

/**
 * Standalone value completion (used without a full schema/query).
 * Throws when a non-null wrapper cannot be satisfied.
 */
export async function completeValue(
  type: TypeRef,
  value: unknown,
): Promise<unknown> {
  const ctx: ExecutionContext = {
    schema: { query: '', types: {} },
    contextValue: {},
    errors: [],
    nextSeq: 0,
  };
  try {
    return await complete(type, value, null, ctx, undefined, undefined);
  } catch (error) {
    if (error === NON_NULL_VIOLATION) {
      throw new Error(
        ctx.errors[0]?.message ?? 'non-null field resolved to null',
      );
    }
    throw error;
  }
}
