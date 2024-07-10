protected void _addMethodMixIns(Class<?> targetClass, AnnotatedMethodMap methods,
        Class<?> mixInCls, AnnotatedMethodMap mixIns)
{
    List<Class<?>> parents = new ArrayList<Class<?>>();
    parents.add(mixInCls);
    ClassUtil.findSuperTypes(mixInCls, targetClass, parents);
    for (Class<?> mixin : parents) {
        for (Method m : mixin.getDeclaredMethods()) {
            if (!_isIncludableMemberMethod(m)) {
                continue;
            }
            AnnotatedMethod am = methods.find(m);
            /* Do we already have a method to augment (from sub-class
             * that will mask this mixIn)? If so, add if visible
             * without masking (no such annotation)
             */
            if (am != null) {
                _addMixUnders(m, am);
            } else {
                // Check if the method already exists in the mixIns map
                AnnotatedMethod existingMethod = mixIns.find(m);
                if (existingMethod == null) {
                    // If not, add the new method to the mixIns map
                    mixIns.add(_constructMethod(m));
                } else {
                    // If the method already exists, augment it with the new method
                    _addMixUnders(m, existingMethod);
                }
            }
        }
    }
}