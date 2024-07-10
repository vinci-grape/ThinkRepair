// Fixed Function
char getMappingCode(final char c) {
    if (!Character.isLetter(c) || Character.toUpperCase(c) < 'A' || Character.toUpperCase(c) > 'Z') {
        return 0;
    }
    return this.soundexMapping[Character.toUpperCase(c) - 'A'];
}