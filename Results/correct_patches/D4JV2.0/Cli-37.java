private boolean isShortOption(String token)
{
    // short options (-S, -SV, -S=V, -SV1=V2, -S1S2)
    if (token.startsWith("-") && token.length() >= 2) {
        String option = token.substring(1);
        // remove "=value"
        if (option.contains("=")) {
            option = option.substring(0, option.indexOf("="));
        }
        // check if the option is valid
        return options.hasShortOption(option);
    }
    return false;
}