public int translate(CharSequence input, int index, Writer out) throws IOException {
    // TODO: Protect from ArrayIndexOutOfBounds
    if (input.charAt(index) == '&' && input.charAt(index + 1) == '#') {
        int start = index + 2;
        boolean isHex = false;

        char firstChar = input.charAt(start);
        if (firstChar == 'x' || firstChar == 'X') {
            start++;
            isHex = true;
        }

        int end = start;
        while (input.charAt(end) != ';') {
            end++;
        }

        int entityValue;
        try {
            if (isHex) {
                entityValue = Integer.parseInt(input.subSequence(start, end).toString(), 16);
            } else {
                entityValue = Integer.parseInt(input.subSequence(start, end).toString(), 10);
            }

            // handle supplementary characters
            final char[] chars = Character.toChars(entityValue);
            out.write(chars); // Fixed Line

            return 2 + (end - start) + (isHex ? 2 : 1); // Fixed Line

        } catch (NumberFormatException nfe) {
            return 0;
        }
    }
    return 0;
}