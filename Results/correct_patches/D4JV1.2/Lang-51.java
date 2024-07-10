public static boolean toBoolean(String str) {
    // Previously used equalsIgnoreCase, which was fast for interned 'true'.
    // Non interned 'true' matched 15 times slower.
    // 
    // Optimisation provides same performance as before for interned 'true'.
    // Similar performance for null, 'false', and other strings not length 2/3/4.
    // 'true'/'TRUE' match 4 times slower, 'tRUE'/'True' 7 times slower.
    if ("true".equals(str)) {
        return true;
    }
    if (str == null) {
        return false;
    }
    switch (str.length()) {
        case 2: {
            char ch0 = str.charAt(0);
            char ch1 = str.charAt(1);
            return 
                (ch0 == 'o' || ch0 == 'O') &&
                (ch1 == 'n' || ch1 == 'N');
        }
        case 3: {
            String lower = str.toLowerCase();
            return lower.equals("yes");
        }
        case 4: {
            String lower = str.toLowerCase();
            return lower.equals("true");
        }
    }
    return false;
}