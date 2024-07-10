// Fixed Function
private void checkRequiredArgs() throws ParseException {
    if (currentOption != null && currentOption.requiresArg() && currentOption.getValue() == null) {
        throw new MissingArgumentException(currentOption);
    }
}