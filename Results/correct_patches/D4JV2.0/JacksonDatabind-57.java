public <T> MappingIterator<T> readValues(byte[] src, int offset, int length)
    throws IOException, JsonProcessingException
{
    if (_dataFormatReaders != null) {
        return _detectBindAndReadValues(_dataFormatReaders.findFormat(src, offset, length), false);
    }
    JsonParser jp = _parserFactory.createParser(src, offset, length); // Fix the argument initialization
    return _bindAndReadValues(_considerFilter(jp, true));
}