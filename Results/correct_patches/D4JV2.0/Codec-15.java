private char getMappingCode(final String str, final int index) {
    // map() throws IllegalArgumentException
    final char mappedChar = this.map(str.charAt(index));
    // HW rule check
    if (index >= 2 && mappedChar != '0') {
        boolean hwRulePassed = false;
        boolean letterFound = false;
        for (int i = index - 1; i >= 0; i--) {
            final char c = str.charAt(i);
            if (c == 'H' || c == 'W') {
                letterFound = true;
            } else if (letterFound && c >= 'A' && c <= 'Z') {
                final char firstCode = this.map(c);
                if (firstCode == mappedChar) {
                    hwRulePassed = true;
                    break;
                }
            } else {
                break;
            }
        }
        if (hwRulePassed) {
            return 0;
        }
    }
    return mappedChar;
}