private boolean toStringEquals(Matcher m, Object arg) {
    if (arg == null) {
        return false;
    }
    return StringDescription.toString(m).equals(arg.toString());
}