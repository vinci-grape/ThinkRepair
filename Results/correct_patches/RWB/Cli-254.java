// Fixed Function
private boolean isJavaProperty(final String token) {
    if (token.isEmpty()) {
        return false;
    }
    
    final String opt = token.substring(0, 1);
    final Option option = options.getOption(opt);

    return option != null && (option.getArgs() >= 2 || option.getArgs() == Option.UNLIMITED_VALUES);
}