// Fixed Function
@Override
void decode(final byte[] data, int offset, final int length, final Context context) {
    if (context.eof || length < 0) {
        context.eof = true;
        if (context.ibitWorkArea != 0) {
            validateTrailingCharacter();
        }
        return;
    }

    final int dataLen = Math.min(data.length - offset, length);
    final int availableChars = (context.ibitWorkArea != 0 ? 1 : 0) + dataLen;

    // small optimisation to short-cut the rest of this method when it is fed byte-by-byte
    if (availableChars == 1 && availableChars == dataLen) {
        // store 1/2 byte for the next invocation of decode, we offset by +1 as empty-value is 0
        context.ibitWorkArea = decodeOctet(data[offset]) + 1;
        return;
    }

    // we must have an even number of chars to decode
    final int charsToProcess = availableChars % BYTES_PER_ENCODED_BLOCK == 0 ? availableChars : availableChars - 1;

    final byte[] buffer = ensureBufferSize(charsToProcess / BYTES_PER_ENCODED_BLOCK, context);

    int result;
    int i = 0;
    if (dataLen < availableChars) {
        // we have 1/2 byte from the previous invocation to decode
        result = (context.ibitWorkArea - 1) << BITS_PER_ENCODED_BYTE;
        result |= decodeOctet(data[offset++]);
        i = 1;

        buffer[context.pos++] = (byte) result;

        // reset to the empty-value for the next invocation!
        context.ibitWorkArea = 0;
    }

    while (i + 1 < charsToProcess) {
        result = decodeOctet(data[offset++]) << BITS_PER_ENCODED_BYTE;
        result |= decodeOctet(data[offset++]);
        i += 2;
        buffer[context.pos++] = (byte) result;
    }

    // we have one char of a hex-pair left over
    if (i < dataLen) {
        // store 1/2 byte for the next invocation of decode, we offset by +1 as the empty-value is 0
        context.ibitWorkArea = decodeOctet(data[offset]) + 1;
    }
}