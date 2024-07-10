boolean matchesLetter() {
    if (isEmpty())
        return false;
    char c = input[pos];
    return Character.isLetter(c); // Use built-in method to check if char is a letter
}