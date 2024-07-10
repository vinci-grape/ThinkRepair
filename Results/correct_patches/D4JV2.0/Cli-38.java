private boolean isShortOption(String token)
{
    // short options (-S, -SV, -S=V, -SV1=V2, -S1S2)
    if (!token.startsWith("-") || token.length() == 1)
    {
        return false;
    }

    // remove leading "-" and "=value"
    int pos = token.indexOf("=");
    String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
    boolean result = options.hasShortOption(optName);

    if (!result && optName.length() > 1) {
        for (int i = 1; i < optName.length(); i++) {
            result = options.hasShortOption(String.valueOf(optName.charAt(i)));
            if (!result) {
                return false;
            }
        }
        return true;
    } else {
        return result;
    }
}