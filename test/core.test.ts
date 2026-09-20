import { describe, expect, it } from 'vitest';
import {
  completeValue,
  execute,
  GraphQLError,
  list,
  named,
  nonNull,
  type DocumentNode,
  type FieldNode,
  type GraphQLSchema,
  type TypeRef,
} from '../src/index.js';

const f = (name: string, alias?: string, selectionSet?: FieldNode[]): FieldNode => ({
  kind: 'field',
  name,
  alias,
  selectionSet,
});

const doc = (...selectionSet: FieldNode[]): DocumentNode => ({
  kind: 'document',
  selectionSet,
});

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    if (signal?.aborted) return reject(Object.assign(new Error('aborted'), { name: 'AbortError' }));
    const t = setTimeout(resolve, ms);
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(t);
        reject(Object.assign(new Error('aborted'), { name: 'AbortError' }));
      },
      { once: true },
    );
  });

// ---------------------------------------------------------------------------
// Schema helpers
// ---------------------------------------------------------------------------

interface ItemDef {
  id: number;
  name?: string | (() => string); // missing -> non-null field resolves null
  fail?: () => never; // resolver throws
}

function itemSchema(itemsType: TypeRef, itemsResolver: () => unknown): GraphQLSchema {
  const Item = {
    kind: 'object' as const,
    name: 'Item',
    fields: {
      id: { type: nonNull(named('Int')) },
      name: {
        type: nonNull(named('String')),
        resolve: (source: ItemDef) =>
          typeof source.name === 'function' ? source.name() : source.name ?? null,
      },
    },
  };
  return {
    query: 'Query',
    types: {
      Query: {
        kind: 'object',
        name: 'Query',
        fields: { items: { type: itemsType, resolve: itemsResolver } },
      },
      Item,
    },
  };
}

const selection = [f('id'), f('name')];
const itemsField = (...sel: FieldNode[]) => f('items', undefined, sel.length ? sel : selection);

const throwName = (code: string, msg = 'boom', extensions: Record<string, unknown> = {}) => () => {
  throw new GraphQLError(msg, { path: undefined, extensions: { code, ...extensions } });
};

// ===========================================================================
// Basic completion (kept from the original suite)
// ===========================================================================

describe('completeValue', () => {
  it('completes a scalar value', async () => {
    expect(await completeValue(named('Int'), 3)).toBe(3);
  });

  it('completes lists of scalars', async () => {
    expect(await completeValue(list(named('Int')), [1, 2, 3])).toEqual([1, 2, 3]);
    expect(await completeValue(nonNull(list(nonNull(named('Int')))), [1, 2])).toEqual([1, 2]);
  });

  it('null element in [T] becomes null', async () => {
    expect(await completeValue(list(named('Int')), [1, null, 3])).toEqual([1, null, 3]);
  });

  it('null element in [T!] throws and nothing escapes the call boundary', async () => {
    await expect(completeValue(list(nonNull(named('Int'))), [1, null, 3])).rejects.toThrow(
      'non-null',
    );
  });

  it('null list in [T]! and [T!]! throws', async () => {
    await expect(completeValue(nonNull(list(named('Int'))), null)).rejects.toThrow('non-null');
    await expect(completeValue(nonNull(list(nonNull(named('Int')))), null)).rejects.toThrow(
      'non-null',
    );
  });

  it('null list in [T] is just null', async () => {
    expect(await completeValue(list(named('Int')), null)).toBeNull();
  });
});

// ===========================================================================
// The four wrapper combinations [T], [T!], [T]!, [T!]!
// ===========================================================================

describe('wrapper matrix: element failure via non-null field', () => {
  const data: ItemDef[] = [
    { id: 1, name: 'a' },
    { id: 2, name: throwName('BAD_NAME') },
    { id: 3, name: 'c' },
  ];

  it('[T]: only the failing element becomes null; siblings survive', async () => {
    const result = await execute(itemSchema(list(named('Item')), () => data), doc(itemsField()));
    expect(result.data).toEqual({
      items: [{ id: 1, name: 'a' }, null, { id: 3, name: 'c' }],
    });
    expect(result.errors).toHaveLength(1);
    expect(result.errors[0].path).toEqual(['items', 1, 'name']);
  });

  it('[T!]: one non-null element fails -> the whole list becomes null', async () => {
    const result = await execute(itemSchema(list(nonNull(named('Item'))), () => data), doc(itemsField()));
    expect(result.data).toEqual({ items: null });
    expect(result.errors).toHaveLength(1);
    expect(result.errors[0].path).toEqual(['items', 1, 'name']);
  });

  it('[T]!: list itself is non-null, but a failing element only nulls the element', async () => {
    const result = await execute(itemSchema(nonNull(list(named('Item'))), () => data), doc(itemsField()));
    expect(result.data).toEqual({
      items: [{ id: 1, name: 'a' }, null, { id: 3, name: 'c' }],
    });
    expect(result.errors[0].path).toEqual(['items', 1, 'name']);
  });

  it('[T!]! behind a nullable parent field: failure bubbles past both ! layers to that field', async () => {
    // items itself is [Item!]! (non-null), so it can never be null; the
    // nearest nullable ancestor is the wrapper field -> wrapper becomes null.
    const schema = itemSchema(nonNull(list(nonNull(named('Item')))), () => data);
    (schema.types.Query as any).fields.wrapper = {
      type: named('Wrapper'),
      resolve: () => ({}),
    };
    schema.types.Wrapper = {
      kind: 'object',
      name: 'Wrapper',
      fields: { items: { type: nonNull(list(nonNull(named('Item')))), resolve: () => data } },
    };
    const result = await execute(
      schema,
      doc(f('wrapper', undefined, [itemsField()])),
    );
    expect(result.data).toEqual({ wrapper: null });
    expect(result.errors[0].path).toEqual(['wrapper', 'items', 1, 'name']);
  });

  it('[T!] behind a nullable parent field: element failure nulls the list, parent survives', async () => {
    const schema = itemSchema(list(nonNull(named('Item'))), () => data);
    (schema.types.Query as any).fields.wrapper = {
      type: named('Wrapper'),
      resolve: () => ({}),
    };
    schema.types.Wrapper = {
      kind: 'object',
      name: 'Wrapper',
      // nullable list [Item!]: element failure nulls the list itself
      fields: { items: { type: list(nonNull(named('Item'))), resolve: () => data } },
    };
    const result = await execute(
      schema,
      doc(f('wrapper', undefined, [itemsField()])),
    );
    expect(result.data).toEqual({ wrapper: { items: null } });
    expect(result.errors[0].path).toEqual(['wrapper', 'items', 1, 'name']);
  });

  it('[T!]! at the top level: failure clears data entirely', async () => {
    const result = await execute(
      itemSchema(nonNull(list(nonNull(named('Item')))), () => data),
      doc(itemsField()),
    );
    expect(result.data).toBeNull();
    expect(result.errors[0].path).toEqual(['items', 1, 'name']);
  });

  it('raw null element in [T!]! nulls the list, then bubbles past the list !', async () => {
    const schema = itemSchema(nonNull(list(nonNull(named('Item')))), () => [
      { id: 1, name: 'a' },
      null,
      { id: 3, name: 'c' },
    ]);
    const result = await execute(schema, doc(itemsField()));
    expect(result.data).toBeNull();
    expect(result.errors[0].path).toEqual(['items', 1]);
  });

  it('raw null element in [T]! is fine: element slot is null, list survives', async () => {
    const schema = itemSchema(nonNull(list(named('Item'))), () => [
      { id: 1, name: 'a' },
      null,
    ]);
    const result = await execute(schema, doc(itemsField()));
    expect(result.data).toEqual({
      items: [{ id: 1, name: 'a' }, null],
    });
    expect(result.errors).toHaveLength(0);
  });

  it('raw null list in [T]! bubbles past the list wrapper to the parent field', async () => {
    const schema = itemSchema(nonNull(list(named('Item'))), () => null);
    const result = await execute(schema, doc(itemsField()));
    expect(result.data).toBeNull();
    expect(result.errors[0].path).toEqual(['items']);
  });

  it('raw null list in [T] is simply null', async () => {
    const schema = itemSchema(list(named('Item')), () => null);
    const result = await execute(schema, doc(itemsField()));
    expect(result.data).toEqual({ items: null });
    expect(result.errors).toHaveLength(0);
  });
});

// ===========================================================================
// Nested lists
// ===========================================================================

describe('nested lists', () => {
  type Matrix = Array<Array<ItemDef | null> | null>;
  const nestedSchema = (itemsType: TypeRef, data: Matrix) => ({
    query: 'Query',
    types: {
      Query: {
        kind: 'object' as const,
        name: 'Query',
        fields: { matrix: { type: itemsType, resolve: () => data } },
      },
      Item: {
        kind: 'object' as const,
        name: 'Item',
        fields: {
          id: { type: nonNull(named('Int')) },
          name: {
            type: nonNull(named('String')),
            resolve: (source: ItemDef) => source.name ?? null,
          },
        },
      },
    },
  });

  const matrixField = f('matrix', undefined, [f('id'), f('name')]);

  it('[[T!]]: failing inner element nulls its (nullable) inner list; outer list survives', async () => {
    const data: Matrix = [
      [{ id: 1, name: 'a' }, { id: 2, name: 'b' }],
      [{ id: 3, name: 'c' }, { id: 4, name: undefined }],
      [{ id: 5, name: 'e' }],
    ];
    const result = await execute(
      nestedSchema(list(list(nonNull(named('Item')))), data),
      doc(matrixField),
    );
    expect(result.data).toEqual({
      matrix: [
        [{ id: 1, name: 'a' }, { id: 2, name: 'b' }],
        null,
        [{ id: 5, name: 'e' }],
      ],
    });
    expect(result.errors[0].path).toEqual(['matrix', 1, 1, 'name']);
  });

  it('[[T!]]!: failing inner element nulls the inner list; the non-null outer list survives', async () => {
    const data: Matrix = [
      [{ id: 1, name: 'a' }],
      [{ id: 2, name: undefined }],
    ];
    const result = await execute(
      nestedSchema(nonNull(list(list(nonNull(named('Item'))))), data),
      doc(matrixField),
    );
    expect(result.data).toEqual({ matrix: [[{ id: 1, name: 'a' }], null] });
  });

  it('[[T!]!]: null inner list bubbles past its non-null element layer and nulls the (nullable) outer list', async () => {
    const data: Matrix = [
      [{ id: 1, name: 'a' }],
      null,
    ];
    // Inner list [Item!]! is non-null and is itself a non-null outer element,
    // so its violation can only stop at the nullable outer list: that whole
    // outer list becomes null (it can never become a single null element).
    const result = await execute(
      nestedSchema(list(nonNull(list(nonNull(named('Item'))))), data),
      doc(matrixField),
    );
    expect(result.data).toEqual({ matrix: null });
    expect(result.errors[0].path).toEqual(['matrix', 1]);
  });

  it('[[T!]!]! at root with null inner list bubbles all the way to data=null', async () => {
    const data: Matrix = [
      [{ id: 1, name: 'a' }],
      null,
    ];
    const result = await execute(
      nestedSchema(nonNull(list(nonNull(list(nonNull(named('Item')))))), data),
      doc(matrixField),
    );
    expect(result.data).toBeNull();
    expect(result.errors[0].path).toEqual(['matrix', 1]);
  });
});

// ===========================================================================
// Aliases
// ===========================================================================

describe('aliases', () => {
  it('errors and data use the alias response key, path points at the failing alias', async () => {
    const schema = itemSchema(list(named('Item')), () => [
      { id: 1, name: 'a' },
      { id: 2, name: throwName('BAD') },
    ]);
    const result = await execute(
      schema,
      doc(f('items', 'renamed', [f('id'), f('name', 'label')])),
    );
    // label is String!, so the whole second element is nulled; the surviving
    // element uses the alias key.
    expect(result.data).toEqual({
      renamed: [{ id: 1, label: 'a' }, null],
    });
    expect(result.errors[0].path).toEqual(['renamed', 1, 'label']);
  });

  it('a nullable failing field under an alias keeps the object and nulls only that key', async () => {
    const schema: GraphQLSchema = {
      query: 'Query',
      types: {
        Query: {
          kind: 'object',
          name: 'Query',
          fields: {
            me: {
              type: named('Me'),
              resolve: () => ({ nickname: null }),
            },
          },
        },
        Me: {
          kind: 'object',
          name: 'Me',
          fields: {
            nickname: {
              type: named('String'),
              resolve: () => {
                throw new GraphQLError('no nick');
              },
            },
          },
        },
      },
    };
    const result = await execute(
      schema,
      doc(f('me', undefined, [f('nickname', 'nick')])),
    );
    expect(result.data).toEqual({ me: { nick: null } });
    expect(result.errors[0].path).toEqual(['me', 'nick']);
  });

  it('the same field twice with different aliases executes independently', async () => {
    const schema = itemSchema(nonNull(list(nonNull(named('Item')))), () => [
      { id: 1, name: 'a' },
      { id: 2, name: throwName('BAD') },
    ]);
    const result = await execute(
      schema,
      doc(
        f('items', 'first', [f('id'), f('name')]),
        f('items', 'second', [f('id'), f('name')]),
      ),
    );
    // Each alias is an independent non-null violation -> data null,
    // both errors retained under their alias paths.
    expect(result.data).toBeNull();
    expect(result.errors.map((e) => e.path)).toEqual([
      ['first', 1, 'name'],
      ['second', 1, 'name'],
    ]);
  });
});

// ===========================================================================
// Multiple simultaneous failures + error ordering
// ===========================================================================

describe('multiple failures and stable ordering', () => {
  it('multiple elements fail in [T]: every failure reported in index order, all sibling data kept', async () => {
    const schema = itemSchema(list(named('Item')), () => [
      { id: 1, name: throwName('E1') },
      { id: 2, name: 'ok' },
      { id: 3, name: throwName('E3') },
      { id: 4, name: throwName('E4') },
    ]);
    const result = await execute(schema, doc(itemsField()));
    expect(result.data).toEqual({
      items: [null, { id: 2, name: 'ok' }, null, null],
    });
    expect(result.errors.map((e) => e.path)).toEqual([
      ['items', 0, 'name'],
      ['items', 2, 'name'],
      ['items', 3, 'name'],
    ]);
  });

  it('multiple elements fail in [T!]!: all errors collected, list nulled once', async () => {
    const schema = itemSchema(nonNull(list(nonNull(named('Item')))), () => [
      { id: 1, name: throwName('E1') },
      { id: 2, name: throwName('E2') },
      { id: 3, name: throwName('E3') },
    ]);
    const result = await execute(schema, doc(itemsField()));
    expect(result.data).toBeNull();
    expect(result.errors.map((e) => e.path)).toEqual([
      ['items', 0, 'name'],
      ['items', 1, 'name'],
      ['items', 2, 'name'],
    ]);
  });

  it('errors across sibling fields sort by query-field order regardless of timing', async () => {
    const schema: GraphQLSchema = {
      query: 'Query',
      types: {
        Query: {
          kind: 'object',
          name: 'Query',
          fields: {
            slow: {
              type: named('String'),
              resolve: async () => {
                await sleep(30);
                throw new GraphQLError('slow failed');
              },
            },
            fast: {
              type: named('String'),
              resolve: async () => {
                await sleep(1);
                throw new GraphQLError('fast failed');
              },
            },
          },
        },
      },
    };
    const result = await execute(schema, doc(f('slow'), f('fast')));
    expect(result.data).toEqual({ slow: null, fast: null });
    expect(result.errors.map((e) => e.path)).toEqual([['slow'], ['fast']]);
  });
});

// ===========================================================================
// Async out-of-order completion
// ===========================================================================

describe('async out-of-order elements', () => {
  it('errors are returned in list-index order when resolvers complete out of order', async () => {
    const schema: GraphQLSchema = {
      query: 'Query',
      types: {
        Query: {
          kind: 'object',
          name: 'Query',
          fields: {
            items: {
              type: list(named('Item')),
              resolve: () => [
                { delay: 40, id: 0 },
                { delay: 1, id: 1 },
                { delay: 20, id: 2 },
                { delay: 5, id: 3 },
              ],
            },
          },
        },
        Item: {
          kind: 'object',
          name: 'Item',
          fields: {
            id: { type: nonNull(named('Int')) },
            name: {
              type: named('String'),
              resolve: async (source: { delay: number; id: number }) => {
                await sleep(source.delay);
                throw new GraphQLError(`late ${source.id}`);
              },
            },
          },
        },
      },
    };
    const result = await execute(schema, doc(itemsField()));
    // Actual completion order would be 1,3,2,0; output order must be 0..3.
    expect(result.data).toEqual({
      items: [
        { id: 0, name: null },
        { id: 1, name: null },
        { id: 2, name: null },
        { id: 3, name: null },
      ],
    });
    expect(result.errors.map((e) => e.path)).toEqual([
      ['items', 0, 'name'],
      ['items', 1, 'name'],
      ['items', 2, 'name'],
      ['items', 3, 'name'],
    ]);
  });

  it('a fast failing non-null element still reports slow elements that already started', async () => {
    let cancelledSibling = false;
    const schema: GraphQLSchema = {
      query: 'Query',
      types: {
        Query: {
          kind: 'object',
          name: 'Query',
          fields: {
            items: {
              type: list(nonNull(named('Item'))),
              resolve: () => [{ id: 0 }, { id: 1 }],
            },
          },
        },
        Item: {
          kind: 'object',
          name: 'Item',
          fields: {
            id: { type: nonNull(named('Int')) },
            name: {
              type: nonNull(named('String')),
              resolve: async (source: { id: number }, _ctx: unknown, info: { signal?: AbortSignal }) => {
                if (source.id === 0) {
                  await sleep(5);
                  throw new GraphQLError('fast failure');
                }
                // Sibling started before the violation: cooperatively notice abort.
                try {
                  await sleep(60, info.signal);
                  return 'done';
                } catch {
                  cancelledSibling = true;
                  throw Object.assign(new Error('aborted'), { name: 'AbortError' });
                }
              },
            },
          },
        },
      },
    };
    const result = await execute(schema, doc(itemsField()));
    expect(result.data).toEqual({ items: null });
    expect(result.errors.map((e) => e.path)).toEqual([['items', 0, 'name']]);
    expect(cancelledSibling).toBe(true); // sibling task was cancellable post-death
  });
});

// ===========================================================================
// Error extensions
// ===========================================================================

describe('error extensions', () => {
  it('preserves extensions on errors pinned to the original field', async () => {
    const schema = itemSchema(list(named('Item')), () => [
      { id: 1, name: throwName('NOT_FOUND', 'missing', { http: 404 }) },
    ]);
    const result = await execute(schema, doc(itemsField()));
    expect(result.errors[0].extensions).toEqual({ code: 'NOT_FOUND', http: 404 });
    expect(result.errors[0].message).toBe('missing');
    expect(result.errors[0].path).toEqual(['items', 0, 'name']);
  });

  it('extensions survive non-null bubbling all the way to data=null', async () => {
    const schema = itemSchema(nonNull(list(nonNull(named('Item')))), () => [
      { id: 1, name: throwName('FATAL', 'deep', { retryable: false }) },
    ]);
    const result = await execute(schema, doc(itemsField()));
    expect(result.data).toBeNull();
    expect(result.errors[0].extensions).toEqual({ code: 'FATAL', retryable: false });
    expect(result.errors[0].path).toEqual(['items', 0, 'name']);
  });

  it('plain Error thrown in a resolver is reported with its path', async () => {
    const schema = itemSchema(list(named('Item')), () => [
      { id: 1, name: () => { throw new Error('ordinary'); } },
    ]);
    const result = await execute(schema, doc(itemsField()));
    expect(result.errors[0].message).toBe('ordinary');
    expect(result.errors[0].path).toEqual(['items', 0, 'name']);
  });
});

// ===========================================================================
// Non-null scalar fields inside objects (the originally reported bug)
// ===========================================================================

describe('object non-null field nulling', () => {
  it('non-null scalar field resolving null bubbles to its nullable object slot', async () => {
    const schema: GraphQLSchema = {
      query: 'Query',
      types: {
        Query: {
          kind: 'object',
          name: 'Query',
          fields: {
            a: { type: named('Widget'), resolve: () => ({ code: null }) },
            b: { type: named('String'), resolve: () => 'kept' },
          },
        },
        Widget: {
          kind: 'object',
          name: 'Widget',
          fields: { code: { type: nonNull(named('String')), resolve: (w: any) => w.code } },
        },
      },
    };
    const result = await execute(schema, doc(f('a', undefined, [f('code')]), f('b')));
    expect(result.data).toEqual({ a: null, b: 'kept' });
    expect(result.errors[0].path).toEqual(['a', 'code']);
  });

  it('non-null object field with failing non-null child nulls the parent field', async () => {
    const schema: GraphQLSchema = {
      query: 'Query',
      types: {
        Query: {
          kind: 'object',
          name: 'Query',
          fields: {
            a: { type: nonNull(named('Widget')), resolve: () => ({ code: null }) },
            b: { type: named('String'), resolve: () => 'kept' },
          },
        },
        Widget: {
          kind: 'object',
          name: 'Widget',
          fields: { code: { type: nonNull(named('String')), resolve: (w: any) => w.code } },
        },
      },
    };
    const result = await execute(schema, doc(f('a', undefined, [f('code')]), f('b')));
    expect(result.data).toBeNull();
    expect(result.errors[0].path).toEqual(['a', 'code']);
  });
});
