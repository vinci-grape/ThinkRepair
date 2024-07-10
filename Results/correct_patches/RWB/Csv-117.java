// Fixed Function
public String get(final Enum<?> e) {
    return get(e == null ? null : e.name());
}