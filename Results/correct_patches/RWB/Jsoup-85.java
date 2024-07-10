// Fixed Function
public static boolean isBooleanAttribute(final String key) {
    for (String attribute : booleanAttributes) {
        if (attribute.equalsIgnoreCase(key)) {
            return true;
        }
    }
    return false;
}