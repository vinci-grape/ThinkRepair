public boolean useForType(JavaType t) {
    switch (_appliesFor) {
        case NON_CONCRETE_AND_ARRAYS:
            while (t.isArrayType()) {
                t = t.getContentType();
            }
            // fall through
        case OBJECT_AND_NON_CONCRETE:
            return t.getRawClass() == Object.class || (!t.isConcrete() && !TreeNode.class.isAssignableFrom(t.getRawClass()));

        case NON_FINAL:
            while (t.isArrayType()) {
                t = t.getContentType();
            }
            return !t.isFinal() && !TreeNode.class.isAssignableFrom(t.getRawClass());

        default:
            return t.getRawClass().equals(Object.class);
    }
}