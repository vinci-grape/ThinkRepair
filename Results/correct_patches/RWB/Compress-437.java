// Fixed Function
@Override
InputStream decode(final String archiveName, final InputStream in, final long uncompressedLength,
        final Coder coder, final byte[] password, final int maxMemoryLimitInKb) throws IOException {
    if (coder.properties == null) {
        throw new IOException("Missing LZMA properties");
    }
    if (coder.properties.length < 1) {
        throw new IOException("LZMA properties too short");
    }
    final byte propsByte = coder.properties[0];
    final int dictSize = getDictionarySize(coder);
    if (dictSize > LZMAInputStream.DICT_SIZE_MAX) {
        throw new IOException("Dictionary larger than 4GiB maximum size used in " + archiveName);
    }
    final int memoryUsageInKb = LZMAInputStream.getMemoryUsage(dictSize, propsByte);
    if (memoryUsageInKb > maxMemoryLimitInKb) {
        throw new MemoryLimitException(memoryUsageInKb, maxMemoryLimitInKb);
    }
    return new LZMAInputStream(in, -1, propsByte, dictSize);
}