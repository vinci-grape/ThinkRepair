// Fixed Function
private static String getMantissa(final String str, final int stopPos) {
    final char firstChar = str.charAt(0);
    final boolean hasSign = firstChar == '-' || firstChar == '+';

    return hasSign && str.length() > 1 ? str.substring(1, stopPos) : str.substring(0, stopPos);
}