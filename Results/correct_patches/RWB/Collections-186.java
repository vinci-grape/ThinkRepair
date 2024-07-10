// Fixed Function
public static <E> List<E> removeAll(final Collection<E> collection, final Collection<?> remove) {
    final List<E> list = new ArrayList<>(collection);
    list.removeAll(remove);
    return list;
}