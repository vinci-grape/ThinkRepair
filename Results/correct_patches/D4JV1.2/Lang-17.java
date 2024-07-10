public final void translate(CharSequence input, Writer out) throws IOException {
    if (out == null) {
        throw new IllegalArgumentException("The Writer must not be null");
    }
    if (input == null) {
        return;
    }
    int pos = 0;
    int len = input.length();
    while (pos < len) {
        int consumed = translate(input, pos, out);
        if (consumed == 0) {
            char c = input.charAt(pos);
            out.write(c);
            pos++; // Increment by 1 for single Java character
        } else {
            for (int pt = 0; pt < consumed; pt++) {
                // Increment by the number of code units in this code point
                pos += Character.charCount(Character.codePointAt(input, pos));
            }
        }
    }
}