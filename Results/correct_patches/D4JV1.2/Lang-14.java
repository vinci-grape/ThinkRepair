public static boolean equals(CharSequence cs1, CharSequence cs2) {
    if (cs1 == cs2) {
        return true;
    }
    if (cs1 == null || cs2 == null) {
        return false;
    }
    if (cs1 instanceof String && cs2 instanceof String) {
        return cs1.equals(cs2);
    }
    return cs1.toString().equals(cs2.toString());
}