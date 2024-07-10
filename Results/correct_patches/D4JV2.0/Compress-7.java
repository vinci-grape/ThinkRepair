public static String parseName(byte[] buffer, final int offset, final int length) {
    StringBuffer result = new StringBuffer(length);
    int          end = offset + length;

    for (int i = offset; i < end; ++i) {
        if (buffer[i] == '\0') { // Fixed Line
            break;
        }
        result.append((char) (buffer[i] & 0xFF)); // Fixed Line
    }

    return result.toString();
}