// GraphQL value-completion core with spec-correct Non-Null propagation.
//
// Model (mirrors the reference implementation):
//
//   1. Every failure — resolver throws, a non-null type resolves to null,
//      scalar coercion fails, a non-iterable reaches a list position — is
//      recorded exactly ONCE, at the location where it happened, carrying
//      that field's full response path (aliases and list indices).
//   2. complete() walks the wrapper chain one frame at a time. A recorded
//      failure travels upward as a NonNullBreach control-flow signal:
//        - a non-null frame re-throws the breach (the value stays invalid);
//        - a nullable frame (named object type or list) absorbs it and
//          completes as null — the breach stops at the first nullable slot.
//      Leaf failures (resolver / scalar) are first judged by the field's own
//      outermost frame, so a nullable failed field is null while a non-null
//      failed field bubbles the SAME original error upward.
//   3. List elements and object fields execute in parallel, each owning an
//      immutable response path plus a query-order ordinal. When a sibling
//      makes its parent container unusable, the container's AbortController
//      aborts every other sibling; errors already collected are kept and
//      returned stably ordered by query field order, never completion order.

export type TypeRef =
  | { kind: 'named'; name: string }
  | { kind: 'list'; ofType: TypeRef }
  | { kind: 'nonNull'; ofType: TypeRef };

/** A response path segment: object response key (alias) or list index. */
export type PathSegment = string | number;

export interface SourceLocation {
  line: number;
  column: number;
}

export interface GraphqlError {
  message: string;
  /** Response path to the field that originally failed (aliases, indices). */
  path: PathSegment[];
  /** Lexicographic key describing query field order, used for stable sort. */
  orderKey: number[];
  locations?: SourceLocation[];
  extensions?: Record<string, unknown>;
}

export interface ResolverInfo {
  /** Immutable response path to this field (alias / index segments). */
  readonly path: PathSegment[];
  readonly fieldName: string;
  /** Response key under which the field appears (its alias, if any). */
  readonly key: string;
  readonly parentTypeName: string;
  readonly type: TypeRef;
  readonly signal: AbortSignal;
}

export type Resolver = (
  source: unknown,
  info: ResolverInfo,
) => unknown | Promise<unknown>;

export interface FieldDef {
  type: TypeRef;
  resolve?: Resolver;
}

export interface ObjectTypeDef {
  kind: 'object';
  fields: Record<string, FieldDef>;
}

export interface ScalarTypeDef {
  kind: 'scalar';
  /** Optional serializer; throwing (or returning undefined) is an error. */
  serialize?: (
    internal: unknown,
    info: { signal: AbortSignal },
  ) => unknown | Promise<unknown>;
}

export type NamedTypeDef = ObjectTypeDef | ScalarTypeDef;

export interface Schema {
  types: Record<string, NamedTypeDef>;
}

export interface SelectionNode {
  /** Field name in the schema. */
  name: string;
  /** Response alias, when one was given; defaults to `name`. */
  alias?: string;
  /** Selection set for object-typed fields. */
  select?: SelectionNode[];
  loc?: SourceLocation;
}

export interface ExecuteOptions {
  schema: Schema;
  /** Root field map (equivalent of the schema's Query root type). */
  rootFields: Record<string, FieldDef>;
  query: SelectionNode[];
  rootValue?: unknown;
  /** Name used in non-null violation messages for root fields. */
  rootTypeName?: string;
  /** Aborting cancels every in-flight resolver; collected errors are kept. */
  signal?: AbortSignal;
}

export interface ExecutionResult {
  data: Record<string, unknown> | null;
  errors?: GraphqlError[];
}

/** A resolver may throw this to attach extensions / a custom message. */
export class ResolverError extends Error {
  readonly extensions?: Record<string, unknown>;
  constructor(message: string, extensions?: Record<string, unknown>) {
    super(message);
    this.name = 'ResolverError';
    this.extensions = extensions;
  }
}

/** Raised internally when execution is cancelled (never a field error). */
export class CancellationError extends Error {
  constructor(message = 'Execution was cancelled') {
    super(message);
    this.name = 'CancellationError';
  }
}

// ---------------------------------------------------------------------------
// Internal control-flow signals
// ---------------------------------------------------------------------------

/**
 * Completion at `error.path` failed; the value must travel up the wrapper
 * chain until a nullable slot absorbs it. The error is already recorded.
 */
class NonNullBreach {
  constructor(readonly error: GraphqlError) {}
}

/** A nullable leaf failed: an outer non-null frame must turn this into a
 *  breach carrying the same error; otherwise the leaf simply becomes null. */
class NullWithError {
  constructor(readonly error: GraphqlError) {}
}

// ---------------------------------------------------------------------------
// Type shorthands
// ---------------------------------------------------------------------------

export const nonNull = (ofType: TypeRef): TypeRef => ({ kind: 'nonNull', ofType });
export const list = (ofType: TypeRef): TypeRef => ({ kind: 'list', ofType });
export const named = (name: string): TypeRef => ({ kind: 'named', name });

export function typeName(type: TypeRef): string {
  switch (type.kind) {
    case 'list':
      return `[${typeName(type.ofType)}]`;
    case 'nonNull':
      return `${typeName(type.ofType)}!`;
    case 'named':
      return type.name;
  }
}

// ---------------------------------------------------------------------------
// Abort plumbing: link a local controller to a parent signal
// ---------------------------------------------------------------------------

interface LinkedAbort {
  controller: AbortController;
  signal: AbortSignal;
}

function createLinkedAbort(parent: AbortSignal): LinkedAbort {
  const controller = new AbortController();
  if (parent.aborted) controller.abort(parent.reason);
  else
    parent.addEventListener(
      'abort',
      () => controller.abort(parent.reason),
      { once: true },
    );
  return { controller, signal: controller.signal };
}

function isCancellation(reason: unknown): boolean {
  return reason instanceof CancellationError;
}

// ---------------------------------------------------------------------------
// Completion context
// ---------------------------------------------------------------------------

interface Ctx {
  schema: Schema;
  errors: GraphqlError[];
  signal: AbortSignal;
  /** Current response path — children fork it with concat(), never mutate. */
  path: readonly PathSegment[];
  /** Lexicographic traversal key (query field order, including indices). */
  order: readonly number[];
  /** Canonical label of the owning field, e.g. "Query.items" / "Item.id". */
  label: string;
  loc?: SourceLocation;
}

const EMPTY_PATH: readonly PathSegment[] = Object.freeze([]);
const EMPTY_ORDER: readonly number[] = Object.freeze([]);

/**
 * Record a failure once at its originating location. After cancellation no
 * new errors are admitted: doomed/cancelled work must not pollute the result.
 */
function recordError(
  ctx: Ctx,
  message: string,
  extensions?: Record<string, unknown>,
): GraphqlError {
  const error: GraphqlError = {
    message,
    path: ctx.path.slice(),
    orderKey: ctx.order.slice(),
    ...(ctx.loc ? { locations: [ctx.loc] } : {}),
    ...(extensions && Object.keys(extensions).length ? { extensions } : {}),
  };
  ctx.errors.push(error);
  return error;
}

function nonNullViolation(ctx: Ctx, type: TypeRef): NonNullBreach {
  if (ctx.signal.aborted) throw new CancellationError();
  const message = ctx.label
    ? `Cannot return null for non-nullable field ${ctx.label}.`
    : `Cannot return null for non-nullable type ${typeName(type)}.`;
  return new NonNullBreach(recordError(ctx, message));
}

/** Convert a thrown resolver/scalar error into a recorded failure. */
function toRecordedError(ctx: Ctx, thrown: unknown): GraphqlError {
  if (ctx.signal.aborted) throw new CancellationError();
  const message = thrown instanceof Error ? thrown.message : String(thrown);
  const extensions =
    thrown != null &&
    typeof thrown === 'object' &&
    'extensions' in thrown
      ? ((thrown as { extensions?: Record<string, unknown> }).extensions ??
        undefined)
      : undefined;
  return recordError(ctx, message, extensions);
}

// ---------------------------------------------------------------------------
// Parallel gate: resolves when every child settles, or as soon as the
// container's own signal aborts. A fatal child stores its breach in the
// shared slot BEFORE aborting siblings, so the caller can re-throw it after
// the gate releases. This also guarantees resolvers that ignore their
// cancellation signal can never hang execution; an external parent abort
// releases the gate with an empty slot (container doomed to null).
// ---------------------------------------------------------------------------

interface FatalSlot {
  breach: NonNullBreach | null;
}

function gate(
  tasks: Array<Promise<unknown>>,
  signal: AbortSignal,
): Promise<void> {
  return new Promise<void>((resolve) => {
    if (signal.aborted || tasks.length === 0) {
      resolve();
      return;
    }
    let pending = tasks.length;
    let settled = false;
    const settle = () => {
      if (!settled) {
        settled = true;
        resolve();
      }
    };
    signal.addEventListener('abort', () => settle(), { once: true });
    for (const task of tasks) {
      task.then(
        () => {
          if (--pending === 0) settle();
        },
        (thrown: unknown) => {
          if (
            !(thrown instanceof NonNullBreach) &&
            !isCancellation(thrown)
          ) {
            settle();
            // Truly unexpected: surface rather than swallow silently.
            queueMicrotask(() => {
              throw thrown;
            });
            return;
          }
          // Breaches/cancellations are handled by the child wrappers.
          if (--pending === 0) settle();
        },
      );
    }
  });
}

// ---------------------------------------------------------------------------
// Value completion — one frame per wrapper type
// ---------------------------------------------------------------------------

function complete(
  type: TypeRef,
  value: unknown,
  ctx: Ctx,
  select: SelectionNode[] | undefined,
): unknown | Promise<unknown> {
  switch (type.kind) {
    // Non-null frame: the inner value must be present. A breach, a tagged
    // leaf failure or an actual null all propagate (the null case recording
    // the violation here, at the field's own path).
    case 'nonNull': {
      const result = complete(type.ofType, value, ctx, select);
      if (isPromise(result)) {
        return result.then((resolved) => checkNonNull(resolved, ctx, type));
      }
      return checkNonNull(result, ctx, type);
    }

    // Nullable list frame: a fatal element breach nulls THIS list. Whether
    // that null then bubbles further is decided by an enclosing frame.
    case 'list': {
      if (value == null) return null;
      return absorbNullable(completeList(type.ofType, value, ctx, select));
    }

    // Named frame. Scalar coercion failures are leaf failures (the wrapping
    // frames judge them); object child breaches are absorbed here because an
    // object type is itself a nullable slot.
    case 'named': {
      if (value == null) return null;
      const def = ctx.schema.types[type.name];
      if (def?.kind === 'scalar') {
        return completeScalar(def, value, ctx);
      }
      if (def?.kind === 'object') {
        // Entering an object type: child field labels are relative to THIS
        // type name, not to the label of the field that returned the object.
        const objectCtx: Ctx = { ...ctx, label: type.name };
        return absorbNullable(
          completeObject(def, value, objectCtx, select ?? []),
        );
      }
      // Unknown named type: opaque scalar passthrough.
      return value;
    }
  }
}

function isPromise(value: unknown): value is Promise<unknown> {
  return (
    value != null &&
    typeof (value as Promise<unknown>).then === 'function'
  );
}

function checkNonNull(
  resolved: unknown,
  ctx: Ctx,
  type: TypeRef,
): unknown {
  if (resolved instanceof NullWithError) {
    // A nullable slot below absorbed a breach: keep carrying the SAME error
    // through us rather than recording a second violation.
    if (ctx.signal.aborted) throw new CancellationError();
    throw new NonNullBreach(resolved.error);
  }
  // A genuine null reached a non-null frame: THIS is the origin of the
  // violation, so record it once at this path.
  if (resolved == null) throw nonNullViolation(ctx, type);
  return resolved;
}

/**
 * Nullable frame: absorb an upward breach. The absorbed value keeps carrying
 * the original error (NullWithError) so an enclosing non-null frame can
 * re-package it; another nullable frame or the root simply sees null.
 */
function absorbNullable(
  result: unknown | Promise<unknown>,
): unknown | Promise<unknown> {
  if (isPromise(result)) {
    return result.then(
      (resolved) => resolved,
      (thrown: unknown) => {
        if (thrown instanceof NonNullBreach) {
          return new NullWithError(thrown.error);
        }
        throw thrown;
      },
    );
  }
  return result;
}

function completeScalar(
  def: ScalarTypeDef,
  value: unknown,
  ctx: Ctx,
): unknown | Promise<unknown> {
  if (!def.serialize) return value;
  try {
    const serialized = def.serialize(value, { signal: ctx.signal });
    if (isPromise(serialized)) {
      return serialized.then(
        (resolved) => {
          if (resolved === undefined) {
            return new NullWithError(
              toRecordedError(
                ctx,
                new Error('A scalar serializer returned undefined.'),
              ),
            );
          }
          return resolved;
        },
        (thrown: unknown) => {
          if (isCancellation(thrown) || ctx.signal.aborted) {
            throw new CancellationError();
          }
          return new NullWithError(toRecordedError(ctx, thrown));
        },
      );
    }
    if (serialized === undefined) {
      return new NullWithError(
        toRecordedError(
          ctx,
          new Error('A scalar serializer returned undefined.'),
        ),
      );
    }
    return serialized;
  } catch (thrown) {
    if (isCancellation(thrown) || ctx.signal.aborted) {
      throw new CancellationError();
    }
    return new NullWithError(toRecordedError(ctx, thrown));
  }
}

function completeList(
  itemType: TypeRef,
  rawValue: unknown,
  ctx: Ctx,
  select: SelectionNode[] | undefined,
): unknown[] | NullWithError | Promise<unknown[]> {
  const iterable = asIterable(rawValue, ctx);
  if (iterable instanceof NullWithError) {
    // Non-iterable value: behave like a failed leaf at this position.
    return iterable;
  }

  const entries: unknown[] = [];
  for (const item of iterable) entries.push(item);

  const results: unknown[] = new Array(entries.length);
  const links = entries.map(() => createLinkedAbort(ctx.signal));
  const fatal: FatalSlot = { breach: null };
  let selfAborted = false;

  const tasks = entries.map((rawItem, i) => {
    const childCtx: Ctx = {
      ...ctx,
      path: ctx.path.concat(i),
      order: ctx.order.concat(i),
      signal: links[i].signal,
    };
    return Promise.resolve(rawItem)
      .then((item) => {
        if (childCtx.signal.aborted) throw new CancellationError();
        return complete(itemType, item, childCtx, select);
      })
      .then(
        (completed) => {
          results[i] =
            completed instanceof NullWithError ? null : completed;
        },
        (thrown: unknown) => {
          if (thrown instanceof NonNullBreach) {
            // A non-null element is unusable: the whole list is unusable.
            // Hold the breach, cancel every sibling, let the gate release;
            // the enclosing wrapper frame decides the nulling boundary.
            fatal.breach ??= thrown;
            selfAborted = true;
            for (const link of links) link.controller.abort();
            return;
          }
          if (isCancellation(thrown)) {
            results[i] = null;
            return;
          }
          throw thrown;
        },
      );
  });

  return gate(tasks, ctx.signal).then(() => {
    // Only propagate a breach caused by OUR child; if our parent aborted us
    // the container is simply doomed and completes with partial results.
    if (selfAborted && fatal.breach) throw fatal.breach;
    return results;
  });
}

function asIterable(
  value: unknown,
  ctx: Ctx,
): Iterable<unknown> | NullWithError {
  if (Array.isArray(value)) return value;
  // Strings are iterable in JS but are NOT valid list values in GraphQL.
  if (
    typeof value !== 'string' &&
    value != null &&
    typeof (value as Iterable<unknown>)[Symbol.iterator] === 'function'
  ) {
    return value as Iterable<unknown>;
  }
  return new NullWithError(
    toRecordedError(
      ctx,
      new Error('Expected an iterable value in a list position.'),
    ),
  );
}

function completeObject(
  def: ObjectTypeDef,
  source: unknown,
  ctx: Ctx,
  selections: SelectionNode[],
):
  | Record<string, unknown>
  | Promise<Record<string, unknown>> {
  const data: Record<string, unknown> = {};
  const links = selections.map(() => createLinkedAbort(ctx.signal));
  const parentTypeName = ctx.label;
  const fatal: FatalSlot = { breach: null };
  let selfAborted = false;

  const tasks = selections.map((selection, ordinal) =>
    executeField(
      def,
      selection,
      ordinal,
      source,
      ctx,
      links,
      parentTypeName,
    ).then(
      (completed) => {
        const key = selection.alias ?? selection.name;
        data[key] = completed instanceof NullWithError ? null : completed;
      },
      (thrown: unknown) => {
        if (thrown instanceof NonNullBreach) {
          // A non-null child field failed: the whole object is unusable.
          // Hold the breach, cancel sibling fields, let the gate release;
          // the enclosing frame (named vs non-null) decides the boundary.
          fatal.breach ??= thrown;
          selfAborted = true;
          for (const link of links) link.controller.abort();
          return;
        }
        if (isCancellation(thrown)) {
          data[selection.alias ?? selection.name] = null;
          return;
        }
        throw thrown;
      },
    ),
  );

  return gate(tasks, ctx.signal).then(() => {
    if (selfAborted && fatal.breach) throw fatal.breach;
    return data;
  });
}

function defaultResolver(source: unknown, fieldName: string): unknown {
  if (source == null) return null;
  return (source as Record<string, unknown>)[fieldName];
}

function executeField(
  parentDef: ObjectTypeDef,
  selection: SelectionNode,
  ordinal: number,
  source: unknown,
  parentCtx: Ctx,
  links: LinkedAbort[],
  parentTypeName: string,
): Promise<unknown> {
  const signal = links[ordinal].signal;
  const key = selection.alias ?? selection.name;
  const fieldDef = parentDef.fields[selection.name];

  return Promise.resolve().then(() => runField());

  function runField(): unknown | Promise<unknown> {
    const fieldCtx: Ctx = {
      ...parentCtx,
      path: parentCtx.path.concat(key),
      order: parentCtx.order.concat(ordinal),
      signal,
      label: `${parentTypeName}.${selection.name}`,
      loc: selection.loc,
    };

    if (!fieldDef) {
      throw nonNullViolation(fieldCtx, {
        kind: 'named',
        name: selection.name,
      });
    }

    const resolve: Resolver =
      fieldDef.resolve ??
      ((src) => defaultResolver(src, selection.name));
    const info: ResolverInfo = {
      path: fieldCtx.path.slice(),
      fieldName: selection.name,
      key,
      parentTypeName,
      type: fieldDef.type,
      signal,
    };

    return Promise.resolve()
      .then(() => {
        if (signal.aborted) throw new CancellationError();
        return resolve(source, info);
      })
      .then(
        (raw) => {
          if (signal.aborted) throw new CancellationError();
          return complete(fieldDef.type, raw, fieldCtx, selection.select);
        },
        (thrown: unknown) => {
          // Resolver failure (or pre-completion abort). A failed nullable
          // field is null; a failed non-null field bubbles the recorded error.
          if (isCancellation(thrown) || signal.aborted) {
            throw new CancellationError();
          }
          const error = toRecordedError(fieldCtx, thrown);
          if (fieldDef.type.kind === 'nonNull') {
            throw new NonNullBreach(error);
          }
          return null;
        },
      );
  }
}

// ---------------------------------------------------------------------------
// Root execution
// ---------------------------------------------------------------------------

export async function execute(options: ExecuteOptions): Promise<ExecutionResult> {
  const {
    schema,
    rootFields,
    query,
    rootValue = {},
    rootTypeName = 'Query',
  } = options;
  const externalSignal =
    options.signal ?? new AbortController().signal;

  const errors: GraphqlError[] = [];
  const rootLink = createLinkedAbort(externalSignal);
  const rootDef: ObjectTypeDef = { kind: 'object', fields: rootFields };
  const rootCtx: Ctx = {
    schema,
    errors,
    signal: rootLink.signal,
    path: EMPTY_PATH,
    order: EMPTY_ORDER,
    label: rootTypeName,
  };

  const data: Record<string, unknown> = {};
  const links = query.map(() => createLinkedAbort(rootLink.signal));
  const fatal: FatalSlot = { breach: null };

  const tasks = query.map((selection, ordinal) =>
    executeField(
      rootDef,
      selection,
      ordinal,
      rootValue,
      rootCtx,
      links,
      rootTypeName,
    ).then(
      (completed) => {
        data[selection.alias ?? selection.name] =
          completed instanceof NullWithError ? null : completed;
      },
      (thrown: unknown) => {
        if (thrown instanceof NonNullBreach) {
          // A top-level non-null field failed: the whole data is null.
          fatal.breach ??= thrown;
          rootLink.controller.abort();
          for (const link of links) link.controller.abort();
          return;
        }
        if (isCancellation(thrown)) {
          data[selection.alias ?? selection.name] = null;
          return;
        }
        throw thrown;
      },
    ),
  );

  await gate(tasks, rootLink.signal);

  // Stable ordering regardless of parallel completion order.
  errors.sort(compareOrderKeys);

  if (fatal.breach) return { data: null, errors };
  return errors.length ? { data, errors } : { data };
}

function compareOrderKeys(a: GraphqlError, b: GraphqlError): number {
  const ka = a.orderKey;
  const kb = b.orderKey;
  const len = Math.min(ka.length, kb.length);
  for (let i = 0; i < len; i++) {
    if (ka[i] !== kb[i]) return ka[i] - kb[i];
  }
  return ka.length - kb.length;
}

// ---------------------------------------------------------------------------
// Standalone value completion (lower-level API)
// ---------------------------------------------------------------------------

export interface CompleteValueOptions {
  schema?: Schema;
  select?: SelectionNode[];
  /** Field label used in non-null violation messages. */
  label?: string;
  signal?: AbortSignal;
}

export interface CompleteValueFailure extends Error {
  graphqlError: GraphqlError;
}

/**
 * Complete a single value against a type.
 *
 * Rejects with a CompleteValueFailure when a non-null violation propagates
 * past the OUTERMOST wrapper (the value itself cannot be returned). When the
 * outermost type is nullable, failures are absorbed and the value resolves
 * to null (matching field semantics); use the full execute() API when the
 * collected error list is needed in that case.
 */
export async function completeValue(
  type: TypeRef,
  value: unknown,
  options?: CompleteValueOptions,
): Promise<unknown> {
  const errors: GraphqlError[] = [];
  const ctx: Ctx = {
    schema: options?.schema ?? { types: {} },
    errors,
    signal: options?.signal ?? new AbortController().signal,
    path: EMPTY_PATH,
    order: EMPTY_ORDER,
    label: options?.label ?? '',
  };
  let result: unknown;
  try {
    result = await Promise.resolve(
      complete(type, value, ctx, options?.select),
    );
  } catch (thrown) {
    if (thrown instanceof NonNullBreach) {
      throw toFailure(thrown.error);
    }
    throw thrown;
  }
  if (result instanceof NullWithError) {
    // Outermost type is nullable: the leaf failure is absorbed here.
    result = null;
  }
  return result;
}

function toFailure(error: GraphqlError): CompleteValueFailure {
  const err = new Error(error.message) as CompleteValueFailure;
  err.graphqlError = { ...error, path: error.path.slice() };
  return err;
}
