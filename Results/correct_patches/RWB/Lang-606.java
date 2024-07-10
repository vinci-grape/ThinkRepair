// Fixed Function
public static Class<?> getRawType(final Type type, final Type assigningType) {
    if (type instanceof Class<?>) {
        // it is raw, no problem
        return (Class<?>) type;
    }

    if (type instanceof ParameterizedType) {
        // simple enough to get the raw type of a ParameterizedType
        return getRawType((ParameterizedType) type);
    }

    if (type instanceof TypeVariable<?>) {
        if (assigningType == null) {
            return null;
        }

        // get the entity declaring this type variable
        final Object genericDeclaration = ((TypeVariable<?>) type).getGenericDeclaration();

        // can't get the raw type of a method- or constructor-declared type variable
        if (!(genericDeclaration instanceof Class<?>)) {
            return null;
        }

        // get the type arguments for the declaring class/interface based on the enclosing type
        final Map<TypeVariable<?>, Type> typeVarAssigns = getTypeArguments(assigningType, (Class<?>) genericDeclaration);

        // enclosingType has to be a subclass (or subinterface) of the declaring type
        if (typeVarAssigns == null) {
            return null;
        }

        // get the argument assigned to this type variable
        final Type typeArgument = typeVarAssigns.get(type);

        if (typeArgument == null) {
            return null;
        }

        // get the argument for this type variable
        return getRawType(typeArgument, assigningType);
    }

    if (type instanceof GenericArrayType) {
        // recursively get the raw type of the component type
        Type componentType = ((GenericArrayType) type).getGenericComponentType();
        Class<?> rawType = getRawType(componentType, assigningType);

        if (rawType != null) {
            return Array.newInstance(rawType, 0).getClass();
        } else {
            return null;
        }
    }

    // (hand-waving) this is not the method you're looking for
    if (type instanceof WildcardType) {
        return null;
    }

    throw new IllegalArgumentException("unknown type: " + type);
}