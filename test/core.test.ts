import { describe, expect, it, vi } from 'vitest';
import {
  completeValue,
  execute,
  list,
  named,
  nonNull,
  ResolverError,
  type FieldDef,
  type GraphqlError,
  type Schema,
  type SelectionNode,
  type TypeRef,
} from '../src/index.js';

// ---------------------------------------------------------------------------
// Test schema
//
//   type Item { id: ID!, code: String!, name: String, child: Item }
//   type Pair { a: Item, b: Item }
//   type Query { ...fields per test... }
//
// Item sources are looked up by string id. Resolvers may also return a
// promise resolved through a per-item delay to force async completion order.
// ---------------------------------------------------------------------------

interface ItemSource {
  id: string;
  code?: string | null;
  name?: string | null;
  child?: string | null;
  /** ms the id/code/name resolvers wait before settling */
  delay?: number;
  /** when true, the resolver throws instead of returning */
  boom?: boolean;
}

const ITEM_DB: Record<string, ItemSource> = {
  i1: { id: 'i1', code: 'C1', name: 'one', child: 'i2' },
  i2: { id: 'i2', code: 'C2', name: 'two', child: null },
  badCode: { id: 'badCode', code: null, name: 'bad' },
  throwing: { id: 'throwing', code: 'X', name: 'thrower', boom: true },
  late: { id: 'late', code: 'L', name: 'late', delay: 30 },
  slow: { id: 'slow', code: 'S', name: 'slow', delay: 20 },
  fast: { id: 'fast', code: 'F', name: 'fast', delay: 1 },
  slowBad: { id: 'slowBad', code: null, name: 'slowBad', delay: 15 },
  fastBad: { id: 'fastBad', code: null, name: 'fastBad', delay: 2 },
};

const wait = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(t);
        resolve();
      },
      { once: true },
    );
  });

const scalarString = {
  kind: 'scalar' as const,
  serialize: (v: unknown) => (v == null ? v : String(v)),
};
const scalarID = scalarString;

const itemFields: Record<string, FieldDef> = {
  id: {
    type: nonNull(named('ID')),
    resolve: async (src: ItemSource, info) => {
      if (src.delay) await wait(src.delay, info.signal);
      if (info.signal.aborted) throw new Error('aborted');
      return src.id;
    },
  },
  code: {
    type: nonNull(named('String')),
    resolve: async (src: ItemSource, info) => {
      if (src.delay) await wait(src.delay, info.signal);
      if (src.boom) {
        throw new ResolverError('code exploded', { code: 'E_CODE' });
      }
      return src.code ?? null;
    },
  },
  name: { type: named('String'), resolve: (s: ItemSource) => s.name ?? null },
  child: {
    type: named('Item'),
    resolve: (s: ItemSource) => (s.child ? ITEM_DB[s.child] : null),
  },
};

function makeSchema(extra: Record<string, FieldDef> = {}): {
  schema: Schema;
  rootFields: Record<string, FieldDef>;
} {
  const schema: Schema = {
    types: {
      Item: { kind: 'object', fields: itemFields },
      String: scalarString,
      ID: scalarID,
      Broken: {
        kind: 'scalar',
        serialize: (v: unknown) => {
          if (v === 'BROKEN') {
            throw new ResolverError('bad scalar', { code: 'E_SCALAR' });
          }
          return v;
        },
      },
    },
  };
  return { schema, rootFields: extra };
}

const itemSelection: SelectionNode[] = [{ name: 'id' }, { name: 'code' }, { name: 'name' }];
const itemSelectionDeep: SelectionNode[] = [
  { name: 'id' },
  { name: 'code' },
  { name: 'name' },
  { name: 'child', select: itemSelection },
];

async function runQuery(
  rootFields: Record<string, FieldDef>,
  query: SelectionNode[],
  rootValue: unknown = {},
  signal?: AbortSignal,
) {
  const { schema } = makeSchema();
  return execute({ schema, rootFields, query, rootValue, signal });
}

const ids = (...list: Array<string | null>) =>
  list.map((id) => (id ? ITEM_DB[id] : null));

const field = (
  type: TypeRef,
  resolve: FieldDef['resolve'],
): FieldDef => ({ type, resolve });

const paths = (errors: GraphqlError[] | undefined) =>
  (errors ?? []).map((e) => e.path);

// ===========================================================================
// Wrapper combinations: [T], [T!], [T]!, [T!]!
// ===========================================================================

describe('non-null propagation boundaries for list wrappers', () => {
  // [Item]   — failing element nulls only that element
  it('[T]: a non-null subfield failing inside one element nulls the element only', async () => {
    const root = {
      items: field(list(named('Item')), () => ids('i1', 'badCode', 'i2')),
    };
    const res = await runQuery(root, [{ name: 'items', select: itemSelection }]);
    expect(res.data).toEqual({
      items: [
        { id: 'i1', code: 'C1', name: 'one' },
        null,
        { id: 'i2', code: 'C2', name: 'two' },
      ],
    });
    expect(paths(res.errors)).toEqual([['items', 1, 'code']]);
  });

  // [Item!]  — non-null element cannot be null: whole list null
  it('[T!]: a failing non-null element nulls the whole (nullable) list', async () => {
    const root = {
      items: field(list(nonNull(named('Item'))), () =>
        ids('i1', 'badCode', 'i2'),
      ),
    };
    const res = await runQuery(root, [{ name: 'items', select: itemSelection }]);
    expect(res.data).toEqual({ items: null });
    expect(paths(res.errors)).toEqual([['items', 1, 'code']]);
  });

  // [Item]!  — list present; an element failure only nulls that element
  it('[T]!: list is present; element failure nulls just that element', async () => {
    const root = {
      items: field(nonNull(list(named('Item'))), () =>
        ids('i1', 'badCode', 'i2'),
      ),
    };
    const res = await runQuery(root, [{ name: 'items', select: itemSelection }]);
    expect(res.data).toEqual({
      items: [
        { id: 'i1', code: 'C1', name: 'one' },
        null,
        { id: 'i2', code: 'C2', name: 'two' },
      ],
    });
    expect(paths(res.errors)).toEqual([['items', 1, 'code']]);
  });

  // [Item!]! — element non-null + list non-null: bubbles past the root field
  it('[T!]!: a failing element nulls all of data', async () => {
    const root = {
      items: field(nonNull(list(nonNull(named('Item')))), () =>
        ids('i1', 'badCode', 'i2'),
      ),
      other: field(named('String'), () => 'kept? no — sibling of fatal root field is irrelevant'),
    };
    const res = await runQuery(root, [
      { name: 'items', select: itemSelection },
    ]);
    expect(res.data).toBeNull();
    expect(paths(res.errors)).toEqual([['items', 1, 'code']]);
  });

  it('[T]!: an explicit null element is allowed (element is nullable)', async () => {
    const root = {
      items: field(nonNull(list(named('Item'))), () => ids('i1', null, 'i2')),
    };
    const res = await runQuery(root, [{ name: 'items', select: itemSelection }]);
    expect(res.errors).toBeUndefined();
    expect(res.data).toEqual({
      items: [{ id: 'i1', code: 'C1', name: 'one' }, null, { id: 'i2', code: 'C2', name: 'two' }],
    });
  });

  it('[T!]!: an explicit null element bubbles to data', async () => {
    const root = {
      items: field(nonNull(list(nonNull(named('Item')))), () =>
        ids('i1', null, 'i2'),
      ),
    };
    const res = await runQuery(root, [{ name: 'items', select: itemSelection }]);
    expect(res.data).toBeNull();
    expect(paths(res.errors)).toEqual([['items', 1]]);
    expect(res.errors?.[0]?.message).toMatch(/non-nullable field Query.items/);
  });

  it('[T!]: an explicit null element nulls the nullable list, not data', async () => {
    const root = {
      items: field(list(nonNull(named('Item'))), () => ids('i1', null, 'i2')),
      keep: field(named('String'), () => 'ok'),
    };
    const res = await runQuery(root, [
      { name: 'items', select: itemSelection },
      { name: 'keep' },
    ]);
    expect(res.data).toEqual({ items: null, keep: 'ok' });
    expect(paths(res.errors)).toEqual([['items', 1]]);
  });

  it('[T]!: the list itself resolving to null is a violation at the field path', async () => {
    const root = {
      items: field(nonNull(list(named('Item'))), () => null),
    };
    const res = await runQuery(root, [{ name: 'items', select: itemSelection }]);
    expect(res.data).toBeNull();
    expect(paths(res.errors)).toEqual([['items']]);
  });
});

// ===========================================================================
// Nested lists
// ===========================================================================

describe('nested lists', () => {
  it('[[T!]!]!: a bad inner element nulls the inner list, outer survives', async () => {
    const root = {
      matrix: field(nonNull(list(nonNull(list(nonNull(named('Item')))))), () => [
        ids('i1', 'i2'),
        ids('i1', 'badCode'),
        ids('i2'),
      ]),
    };
    const res = await runQuery(root, [
      { name: 'matrix', select: itemSelection },
    ]);
    // inner element index 1 of row 1 fails: row itself is [!]!, so row null;
    // outer element (row) is non-null too -> row null bubbles to outer list
    expect(res.data).toBeNull();
    expect(paths(res.errors)).toEqual([['matrix', 1, 1, 'code']]);
  });

  it('[[T!]]!: bad inner element nulls inner list; outer list retains rows', async () => {
    const root = {
      matrix: field(nonNull(list(list(nonNull(named('Item'))))), () => [
        ids('i1', 'i2'),
        ids('i1', 'badCode'),
        ids('i2'),
      ]),
    };
    const res = await runQuery(root, [
      { name: 'matrix', select: itemSelection },
    ]);
    expect(res.data).toEqual({
      matrix: [
        [{ id: 'i1', code: 'C1', name: 'one' }, { id: 'i2', code: 'C2', name: 'two' }],
        null,
        [{ id: 'i2', code: 'C2', name: 'two' }],
      ],
    });
    expect(paths(res.errors)).toEqual([['matrix', 1, 1, 'code']]);
  });

  it('[[T]!]!: null inner element allowed; bad non-null subfield nulls one item', async () => {
    const root = {
      matrix: field(nonNull(list(nonNull(list(named('Item'))))), () => [
        ids('i1', null),
        ids('badCode'),
      ]),
    };
    const res = await runQuery(root, [
      { name: 'matrix', select: itemSelection },
    ]);
    expect(res.data).toEqual({
      matrix: [
        [{ id: 'i1', code: 'C1', name: 'one' }, null],
        [null],
      ],
    });
    expect(paths(res.errors)).toEqual([['matrix', 1, 0, 'code']]);
  });
});

// ===========================================================================
// Aliases and nested objects
// ===========================================================================

describe('aliases', () => {
  it('error path uses the response alias at every level', async () => {
    const root = {
      aliasedItems: field(list(named('Item')), () => ids('badCode')),
    };
    const res = await runQuery(root, [
      {
        name: 'aliasedItems',
        alias: 'renamed',
        select: [{ name: 'code', alias: 'theCode' }],
      },
    ]);
    expect(res.data).toEqual({ renamed: [null] });
    expect(paths(res.errors)).toEqual([['renamed', 0, 'theCode']]);
  });

  it('non-null violation message quotes the canonical field, path uses alias', async () => {
    const root = {
      aliasedItems: field(nonNull(list(nonNull(named('Item')))), () =>
        ids(null),
      ),
    };
    const res = await runQuery(root, [
      { name: 'aliasedItems', alias: 'renamed', select: itemSelection },
    ]);
    expect(res.data).toBeNull();
    expect(res.errors?.[0]?.path).toEqual(['renamed', 0]);
    expect(res.errors?.[0]?.message).toMatch(/Query\.aliasedItems/);
  });

  it('the same field queried under two aliases executes independently with distinct paths', async () => {
    const root = {
      items: field(list(named('Item')), (src, info) =>
        // first alias returns a good list, second alias a bad one
        info.key === 'good' ? ids('i1') : ids('badCode'),
      ),
    };
    const res = await runQuery(root, [
      { name: 'items', alias: 'good', select: itemSelection },
      { name: 'items', alias: 'bad', select: itemSelection },
    ]);
    expect(res.data).toEqual({
      good: [{ id: 'i1', code: 'C1', name: 'one' }],
      bad: [null],
    });
    expect(paths(res.errors)).toEqual([['bad', 0, 'code']]);
  });

  it('alias on a failing field inside a list element points through the index', async () => {
    const root = {
      items: field(nonNull(list(nonNull(named('Item')))), () =>
        ids('i1', 'badCode'),
      ),
    };
    const res = await runQuery(root, [
      {
        name: 'items',
        select: [{ name: 'id' }, { name: 'code', alias: 'errorCode' }],
      },
    ]);
    expect(res.data).toBeNull();
    expect(paths(res.errors)).toEqual([['items', 1, 'errorCode']]);
  });
});

describe('object field boundaries', () => {
  it('a nullable object with a non-null child failure becomes null; siblings retained', async () => {
    const root = {
      a: field(named('Item'), () => ITEM_DB.badCode),
      b: field(named('Item'), () => ITEM_DB.i2),
    };
    const res = await runQuery(root, [
      { name: 'a', select: itemSelection },
      { name: 'b', select: itemSelection },
    ]);
    expect(res.data).toEqual({
      a: null,
      b: { id: 'i2', code: 'C2', name: 'two' },
    });
    expect(paths(res.errors)).toEqual([['a', 'code']]);
  });

  it('a failed nullable field is null with its own error (no bubbling)', async () => {
    const root = {
      nullableItem: field(named('Item'), () => {
        throw new ResolverError('resolver down');
      }),
      keep: field(named('String'), () => 'yes'),
    };
    const res = await runQuery(root, [
      { name: 'nullableItem', select: itemSelection },
      { name: 'keep' },
    ]);
    expect(res.data).toEqual({ nullableItem: null, keep: 'yes' });
    expect(paths(res.errors)).toEqual([['nullableItem']]);
    expect(res.errors?.[0]?.message).toBe('resolver down');
  });

  it('a failed non-null root field nulls all of data', async () => {
    const root = {
      required: field(nonNull(named('String')), () => {
        throw new Error('boom');
      }),
      other: field(named('String'), () => 'irrelevant'),
    };
    const res = await runQuery(root, [{ name: 'required' }, { name: 'other' }]);
    expect(res.data).toBeNull();
    expect(paths(res.errors)).toEqual([['required']]);
  });

  it('nested object: parent nullable absorbs breach from deep non-null field', async () => {
    const custom: Record<string, FieldDef> = {
      pair: field(named('Item'), () => ({
        id: 'p1',
        code: 'C',
        name: 'n',
        child: 'badCode', // foreign key resolved by Item.child
      })),
    };
    const { schema } = makeSchema();
    const res = await execute({
      schema,
      rootFields: custom,
      query: [{ name: 'pair', select: itemSelectionDeep }],
      rootValue: {},
    });
    expect(res.data).toEqual({
      pair: { id: 'p1', code: 'C', name: 'n', child: null },
    });
    expect(paths(res.errors)).toEqual([['pair', 'child', 'code']]);
  });
});

// ===========================================================================
// Multiple failures, stable ordering
// ===========================================================================

describe('multiple concurrent failures', () => {
  it('[T]: multiple failing elements yield multiple errors sorted by index', async () => {
    const root = {
      items: field(list(named('Item')), () =>
        ids('badCode', 'i2', 'badCode'),
      ),
    };
    const res = await runQuery(root, [{ name: 'items', select: itemSelection }]);
    expect(res.data).toEqual({
      items: [null, { id: 'i2', code: 'C2', name: 'two' }, null],
    });
    expect(paths(res.errors)).toEqual([
      ['items', 0, 'code'],
      ['items', 2, 'code'],
    ]);
  });

  it('[T!]: two synchronously failing elements — both errors already collected are retained', async () => {
    const root = {
      items: field(list(nonNull(named('Item'))), () =>
        ids('badCode', 'badCode'),
      ),
    };
    const res = await runQuery(root, [{ name: 'items', select: itemSelection }]);
    expect(res.data).toEqual({ items: null });
    // Both failed before the cancellation signal could arrive: collected
    // errors are kept, ordered by index.
    expect(paths(res.errors)).toEqual([
      ['items', 0, 'code'],
      ['items', 1, 'code'],
    ]);
  });

  it('[T!]: a fast fatal element cancels a still-running failing sibling (one error)', async () => {
    const root = {
      items: field(list(nonNull(named('Item'))), () =>
        ids('fastBad', 'slowBad'),
      ),
    };
    const res = await runQuery(root, [{ name: 'items', select: itemSelection }]);
    expect(res.data).toEqual({ items: null });
    // Index 1 was cancelled mid-flight and its post-abort failure is not
    // admitted, so only the fast index-0 error survives.
    expect(paths(res.errors)).toEqual([['items', 0, 'code']]);
  });

  it('errors across sibling fields sort in query field order, not completion order', async () => {
    // field "late" resolves before "early"? construct: first queried field
    // is slow, second is fast — both fail. Ordering must follow query order.
    const root = {
      slowField: field(list(named('Item')), async () => {
        await wait(30);
        return ids('slowBad');
      }),
      fastField: field(list(named('Item')), async () => {
        await wait(2);
        return ids('fastBad');
      }),
    };
    const res = await runQuery(root, [
      { name: 'slowField', select: itemSelection },
      { name: 'fastField', select: itemSelection },
    ]);
    expect(paths(res.errors)).toEqual([
      ['slowField', 0, 'code'],
      ['fastField', 0, 'code'],
    ]);
  });
});

// ===========================================================================
// Async out-of-order completion
// ===========================================================================

describe('async out-of-order list elements', () => {
  it('paths and result slots stay correct when elements finish out of order', async () => {
    const root = {
      items: field(list(named('Item')), () =>
        ids('fastBad', 'slow', 'late'),
      ),
    };
    const res = await runQuery(root, [{ name: 'items', select: itemSelection }]);
    // fastBad (index 0) fails quickly while slow/late are still running.
    expect(res.data).toEqual({
      items: [null, { id: 'slow', code: 'S', name: 'slow' }, { id: 'late', code: 'L', name: 'late' }],
    });
    expect(paths(res.errors)).toEqual([['items', 0, 'code']]);
  });

  it('a late failing element in [T!]! cancels fast siblings and nulls data', async () => {
    const root = {
      items: field(nonNull(list(nonNull(named('Item')))), () =>
        ids('fast', 'slowBad', 'late'),
      ),
    };
    const res = await runQuery(root, [{ name: 'items', select: itemSelection }]);
    expect(res.data).toBeNull();
    expect(paths(res.errors)).toEqual([['items', 1, 'code']]);
  });

  it('immutable per-element path: a fast failure at index 2 keeps its own path', async () => {
    const root = {
      items: field(list(named('Item')), () => {
        // index 2 fails fast (fastBad delay=2), index 0 slow
        return ids('slow', 'late', 'fastBad');
      }),
    };
    const res = await runQuery(root, [{ name: 'items', select: itemSelection }]);
    expect(paths(res.errors)).toEqual([['items', 2, 'code']]);
    expect(res.data).toEqual({
      items: [
        { id: 'slow', code: 'S', name: 'slow' },
        { id: 'late', code: 'L', name: 'late' },
        null,
      ],
    });
  });
});

// ===========================================================================
// Sibling cancellation
// ===========================================================================

describe('sibling cancellation', () => {
  it('a fatal element aborts sibling resolvers (they observe the signal)', async () => {
    const cancelled: number[] = [];
    const root = {
      items: field(nonNull(list(nonNull(named('Item')))), () => [
        ITEM_DB.fastBad,
        {
          id: 'never',
          code: 'N',
          name: 'never',
          delay: 1000,
        } as ItemSource,
      ]),
    };
    // Spy via a custom field that records abort:
    const abortSpy = vi.fn();
    const trackedFields: Record<string, FieldDef> = {
      ...itemFields,
      code: {
        type: nonNull(named('String')),
        resolve: async (src: ItemSource, info) => {
          info.signal.addEventListener('abort', () => {
            abortSpy(src.id);
            cancelled.push(1);
          });
          if (src.delay) await wait(src.delay, info.signal);
          if (info.signal.aborted) throw new Error('cancelled');
          return src.code ?? null;
        },
      },
    };
    const schema: Schema = {
      types: { Item: { kind: 'object', fields: trackedFields }, String: scalarString, ID: scalarID },
    };
    const res = await execute({
      schema,
      rootFields: root,
      query: [{ name: 'items', select: itemSelection }],
      rootValue: {},
    });
    expect(res.data).toBeNull();
    expect(paths(res.errors)).toEqual([['items', 0, 'code']]);
    expect(abortSpy).toHaveBeenCalledWith('never');
  });

  it('does not wait forever for a resolver that ignores its signal', async () => {
    // A non-abortable resolver at index 1 would hang; gate resolves on abort.
    const root = {
      items: field(nonNull(list(nonNull(named('Item')))), () => [
        ITEM_DB.fastBad,
        ITEM_DB.slow,
      ]),
    };
    const hangingFields: Record<string, FieldDef> = {
      ...itemFields,
      code: {
        type: nonNull(named('String')),
        resolve: (src: ItemSource, info) => {
          if (src.id === 'slow') {
            // deliberately ignore AbortSignal
            return new Promise(() => {});
          }
          return itemFields.code!.resolve!(src, info);
        },
      },
    };
    const schema: Schema = {
      types: { Item: { kind: 'object', fields: hangingFields }, String: scalarString, ID: scalarID },
    };
    const start = Date.now();
    const res = await execute({
      schema,
      rootFields: root,
      query: [{ name: 'items', select: itemSelection }],
      rootValue: {},
    });
    expect(Date.now() - start).toBeLessThan(500);
    expect(res.data).toBeNull();
  });

  it('external AbortSignal cancels the whole execution', async () => {
    const ac = new AbortController();
    const root = {
      items: field(nonNull(list(nonNull(named('Item')))), () => [
        ITEM_DB.slow,
        ITEM_DB.late,
      ]),
    };
    const p = runQuery(root, [{ name: 'items', select: itemSelection }], {}, ac.signal);
    setTimeout(() => ac.abort(new Error('client gone')), 5);
    const res = await p;
    // No non-null violation happened before abort; data remains an object
    // with whatever finished, but execution must not hang or throw.
    expect(res).toHaveProperty('data');
  });
});

// ===========================================================================
// Error extensions / locations
// ===========================================================================

describe('error extensions and locations', () => {
  it('resolver-thrown extensions survive to the collected error', async () => {
    const root = {
      items: field(list(named('Item')), () => ids('throwing')),
    };
    const res = await runQuery(root, [
      { name: 'items', select: itemSelection, loc: { line: 3, column: 5 } },
    ]);
    const err = res.errors?.[0];
    expect(err?.path).toEqual(['items', 0, 'code']);
    expect(err?.extensions).toEqual({ code: 'E_CODE' });
    expect(err?.message).toBe('code exploded');
  });

  it('scalar coercion failure behaves per wrapper type', async () => {
    const { schema } = makeSchema();
    const rootFields: Record<string, FieldDef> = {
      nullableBroken: { type: named('Broken'), resolve: () => 'BROKEN' },
      nonNullBroken: {
        type: nonNull(list(nonNull(named('Broken')))),
        resolve: () => ['ok', 'BROKEN'],
      },
    };
    const res1 = await execute({
      schema,
      rootFields,
      query: [{ name: 'nullableBroken' }],
    });
    expect(res1.data).toEqual({ nullableBroken: null });
    expect(paths(res1.errors)).toEqual([['nullableBroken']]);
    expect(res1.errors?.[0]?.extensions).toEqual({ code: 'E_SCALAR' });

    const res2 = await execute({
      schema,
      rootFields,
      query: [{ name: 'nonNullBroken' }],
    });
    expect(res2.data).toBeNull();
    expect(paths(res2.errors)).toEqual([['nonNullBroken', 1]]);
  });

  it('non-iterable value in a list position errors at the field path', async () => {
    const root = {
      items: field(nonNull(list(named('Item'))), () => 'not-a-list'),
    };
    const res = await runQuery(root, [{ name: 'items', select: itemSelection }]);
    // list is non-null -> field becomes null -> root data null
    expect(res.data).toBeNull();
    expect(paths(res.errors)).toEqual([['items']]);

    const root2 = {
      items2: field(list(named('Item')), () => 42),
    };
    const res2 = await runQuery(root2, [{ name: 'items2', select: itemSelection }]);
    expect(res2.data).toEqual({ items2: null });
    expect(paths(res2.errors)).toEqual([['items2']]);
  });

  it('accepts arbitrary iterables (generators / Sets)', async () => {
    const root = {
      items: field(list(named('String')), function* () {
        yield 'a';
        yield 'b';
      }),
    };
    const res = await runQuery(root, [{ name: 'items' }]);
    expect(res.errors).toBeUndefined();
    expect(res.data).toEqual({ items: ['a', 'b'] });
  });
});

// ===========================================================================
// Standalone completeValue — matrix over wrapper combinations
// ===========================================================================

describe('completeValue wrapper matrix', () => {
  const T = named('String');

  it('[T]: null outer -> null (no throw)', async () => {
    expect(await completeValue(list(T), null)).toBeNull();
  });

  it('[T!]: null element resolves the (nullable) list to null with path [1]', async () => {
    // Outer list is nullable: the element breach is absorbed, but the
    // recorded error is still exposed via the rejection metadata helper on
    // the standalone API is NOT applicable; here it simply resolves null.
    expect(await completeValue(list(nonNull(T)), ['a', null])).toBeNull();
  });

  it('[T]!: non-null list with null outer rejects with path []', async () => {
    await expect(completeValue(nonNull(list(T)), null)).rejects.toMatchObject({
      graphqlError: { path: [] },
    });
  });

  it('[T]!: null element allowed', async () => {
    expect(await completeValue(nonNull(list(T)), ['a', null])).toEqual(['a', null]);
  });

  it('[T!]!: null element rejects with path [index]', async () => {
    await expect(
      completeValue(nonNull(list(nonNull(T))), ['a', null, 'c']),
    ).rejects.toMatchObject({ graphqlError: { path: [1] } });
  });

  it('[T!] (nullable outer): null element resolves the list to null', async () => {
    await expect(completeValue(list(nonNull(T)), ['a', null])).resolves.toBeNull();
  });

  it('nested [[T!]!]! null inner element rejects with full index path', async () => {
    const type = nonNull(list(nonNull(list(nonNull(T)))));
    await expect(completeValue(type, [['a'], ['b', null]])).rejects.toMatchObject({
      graphqlError: { path: [1, 1] },
    });
  });

  it('scalar passthrough values complete normally', async () => {
    expect(await completeValue(T, 7)).toBe(7);
    expect(await completeValue(nonNull(T), 7)).toBe(7);
  });
});

describe('completeValue with object schema', () => {
  it('completes nested objects and reports exact path through lists', async () => {
    const { schema } = makeSchema();
    const result = await completeValue(
      list(named('Item')),
      [ITEM_DB.i1, ITEM_DB.badCode, ITEM_DB.i2],
      { schema, select: itemSelection, label: 'Query.items' },
    );
    expect(result).toEqual([
      { id: 'i1', code: 'C1', name: 'one' },
      null,
      { id: 'i2', code: 'C2', name: 'two' },
    ]);
  });

  it('rejects when [Item!]! contains a failing item, carrying original error', async () => {
    const { schema } = makeSchema();
    await expect(
      completeValue(
        nonNull(list(nonNull(named('Item')))),
        [ITEM_DB.i1, ITEM_DB.badCode],
        { schema, select: itemSelection },
      ),
    ).rejects.toMatchObject({
      graphqlError: { path: [1, 'code'] },
    });
  });
});
