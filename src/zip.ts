// --- Minimal store-only (uncompressed) ZIP writer — no dependency ---
const enc = new TextEncoder();
const dec = new TextDecoder("utf-8", { fatal: true });
const u16 = (n: number) => Uint8Array.of(n & 0xff, (n >> 8) & 0xff);
const u32 = (n: number) => Uint8Array.of(n & 0xff, (n >> 8) & 0xff, (n >> 16) & 0xff, (n >>> 24) & 0xff);
const concat = (parts: Uint8Array[]) => {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
};
const CRC_TABLE = Uint32Array.from({ length: 256 }, (_, i) => {
  let c = i;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
function crc32(data: Uint8Array) {
  let c = 0xffffffff;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
export function zip(files: { name: string; data: Uint8Array }[]) {
  if (files.length > 0xfffe) {
    throw new Error("ZIP has too many files for the non-ZIP64 format");
  }
  const local: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  const names = new Set<string>();
  let offset = 0;
  for (const f of files) {
    if (!f.name || names.has(f.name)) {
      throw new Error(`ZIP has a duplicate or empty file name "${f.name}"`);
    }
    names.add(f.name);
    const nb = enc.encode(f.name);
    if (nb.length > 0xffff) throw new Error("ZIP file name is too long");
    const crc = crc32(f.data);
    const sz = f.data.length;
    if (sz > 0xffffffff || offset > 0xffffffff) {
      throw new Error("ZIP file is too large for the non-ZIP64 format");
    }
    const header = concat([u32(0x04034b50), u16(20), u16(0), u16(0), u16(0), u16(0),
      u32(crc), u32(sz), u32(sz), u16(nb.length), u16(0), nb]);
    local.push(header, f.data);
    central.push(concat([u32(0x02014b50), u16(20), u16(20), u16(0), u16(0), u16(0), u16(0),
      u32(crc), u32(sz), u32(sz), u16(nb.length), u16(0), u16(0), u16(0), u16(0), u32(0), u32(offset), nb]));
    offset += header.length + sz;
  }
  if (offset > 0xffffffff) {
    throw new Error("ZIP archive is too large for the non-ZIP64 format");
  }
  const cd = concat(central);
  const end = concat([u32(0x06054b50), u16(0), u16(0), u16(files.length), u16(files.length),
    u32(cd.length), u32(offset), u16(0)]);
  return concat([...local, cd, end]);
}

const requireRange = (
  data: Uint8Array,
  offset: number,
  length: number,
  context: string,
) => {
  if (offset < 0 || length < 0 || offset + length > data.length) {
    throw new Error(`Invalid ZIP: ${context} is outside the archive`);
  }
};

const readU16 = (view: DataView, offset: number) => {
  if (offset < 0 || offset + 2 > view.byteLength) {
    throw new Error("Invalid ZIP: 16-bit field is outside the archive");
  }
  return view.getUint16(offset, true);
};

const readU32 = (view: DataView, offset: number) => {
  if (offset < 0 || offset + 4 > view.byteLength) {
    throw new Error("Invalid ZIP: 32-bit field is outside the archive");
  }
  return view.getUint32(offset, true);
};

/**
 * Reads the store-only ZIP files produced by zip().
 *
 * This deliberately rejects compressed, encrypted, multi-disk, and ZIP64
 * archives. Dataset import needs one small, auditable format rather than a
 * general archive extractor.
 */
export function unzip(data: Uint8Array): Map<string, Uint8Array> {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const minimumEocdLength = 22;
  const maximumCommentLength = 0xffff;
  const searchStart = Math.max(
    0,
    data.length - minimumEocdLength - maximumCommentLength,
  );
  let eocdOffset = -1;

  for (let offset = data.length - minimumEocdLength; offset >= searchStart; offset--) {
    if (readU32(view, offset) === 0x06054b50) {
      eocdOffset = offset;
      break;
    }
  }
  if (eocdOffset < 0) throw new Error("Invalid ZIP: end record not found");

  const diskNumber = readU16(view, eocdOffset + 4);
  const centralDisk = readU16(view, eocdOffset + 6);
  const diskEntries = readU16(view, eocdOffset + 8);
  const entryCount = readU16(view, eocdOffset + 10);
  const centralSize = readU32(view, eocdOffset + 12);
  const centralOffset = readU32(view, eocdOffset + 16);
  const commentLength = readU16(view, eocdOffset + 20);

  if (diskNumber || centralDisk || diskEntries !== entryCount) {
    throw new Error("Unsupported ZIP: multi-disk archives are not accepted");
  }
  if (
    entryCount === 0xffff ||
    centralSize === 0xffffffff ||
    centralOffset === 0xffffffff
  ) {
    throw new Error("Unsupported ZIP: ZIP64 archives are not accepted");
  }
  requireRange(data, eocdOffset + minimumEocdLength, commentLength, "comment");
  if (eocdOffset + minimumEocdLength + commentLength !== data.length) {
    throw new Error("Invalid ZIP: trailing data after the end record");
  }
  requireRange(data, centralOffset, centralSize, "central directory");
  if (centralOffset + centralSize !== eocdOffset) {
    throw new Error("Invalid ZIP: central directory does not precede the end record");
  }

  const files = new Map<string, Uint8Array>();
  let cursor = centralOffset;
  const centralEnd = centralOffset + centralSize;

  for (let index = 0; index < entryCount; index++) {
    requireRange(data, cursor, 46, "central directory entry");
    if (readU32(view, cursor) !== 0x02014b50) {
      throw new Error("Invalid ZIP: central directory signature mismatch");
    }

    const flags = readU16(view, cursor + 8);
    const method = readU16(view, cursor + 10);
    const expectedCrc = readU32(view, cursor + 16);
    const compressedSize = readU32(view, cursor + 20);
    const uncompressedSize = readU32(view, cursor + 24);
    const nameLength = readU16(view, cursor + 28);
    const extraLength = readU16(view, cursor + 30);
    const entryCommentLength = readU16(view, cursor + 32);
    const localOffset = readU32(view, cursor + 42);
    const entryLength =
      46 + nameLength + extraLength + entryCommentLength;

    requireRange(data, cursor, entryLength, "central directory entry");
    if (flags & 0x0001) {
      throw new Error("Unsupported ZIP: encrypted files are not accepted");
    }
    if (method !== 0 || compressedSize !== uncompressedSize) {
      throw new Error(
        "Unsupported ZIP: only uncompressed files are accepted",
      );
    }
    if (
      compressedSize === 0xffffffff ||
      uncompressedSize === 0xffffffff ||
      localOffset === 0xffffffff
    ) {
      throw new Error("Unsupported ZIP: ZIP64 entries are not accepted");
    }

    const name = dec.decode(
      data.subarray(cursor + 46, cursor + 46 + nameLength),
    );
    if (!name || files.has(name)) {
      throw new Error(`Invalid ZIP: duplicate or empty file name "${name}"`);
    }

    requireRange(data, localOffset, 30, `local header for ${name}`);
    if (readU32(view, localOffset) !== 0x04034b50) {
      throw new Error(`Invalid ZIP: local header missing for ${name}`);
    }
    const localNameLength = readU16(view, localOffset + 26);
    const localExtraLength = readU16(view, localOffset + 28);
    requireRange(
      data,
      localOffset + 30,
      localNameLength + localExtraLength,
      `local name and extra data for ${name}`,
    );
    const localName = dec.decode(
      data.subarray(localOffset + 30, localOffset + 30 + localNameLength),
    );
    if (localName !== name) {
      throw new Error(`Invalid ZIP: local file name mismatch for ${name}`);
    }
    const payloadOffset =
      localOffset + 30 + localNameLength + localExtraLength;
    requireRange(data, payloadOffset, compressedSize, `payload for ${name}`);
    const payload = data.slice(
      payloadOffset,
      payloadOffset + compressedSize,
    );
    if (crc32(payload) !== expectedCrc) {
      throw new Error(`Invalid ZIP: CRC mismatch for ${name}`);
    }
    files.set(name, payload);
    cursor += entryLength;
  }

  if (cursor !== centralEnd) {
    throw new Error("Invalid ZIP: central directory length mismatch");
  }
  return files;
}
// self-check: CRC-32 of "123456789" is the standard 0xCBF43926 test vector
if (import.meta.env.DEV && crc32(enc.encode("123456789")) !== 0xcbf43926) throw new Error("crc32 broken");
