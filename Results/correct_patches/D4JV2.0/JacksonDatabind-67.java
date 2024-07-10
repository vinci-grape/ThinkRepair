public KeyDeserializer createKeyDeserializer(DeserializationContext ctxt,
        JavaType type)
    throws JsonMappingException
{
    final DeserializationConfig config = ctxt.getConfig();
    KeyDeserializer deser = null;
    if (_factoryConfig.hasKeyDeserializers()) {
        BeanDescription beanDesc = config.introspectClassAnnotations(type.getRawClass());
        for (KeyDeserializers d : _factoryConfig.keyDeserializers()) {
            KeyDeserializer tempDeser = d.findKeyDeserializer(type, config, beanDesc);
            if (tempDeser != null) {
                deser = tempDeser;
                break;
            }
        }
    }
    // the only non-standard thing is this:
    if (deser == null) {
        if (type.isEnumType()) {
            deser = _createEnumKeyDeserializer(ctxt, type);
        } else {
            deser = StdKeyDeserializers.findStringBasedKeyDeserializer(config, type);
        }
    }
    // and then post-processing
    if (deser != null && _factoryConfig.hasDeserializerModifiers()) {
        for (BeanDeserializerModifier mod : _factoryConfig.deserializerModifiers()) {
            deser = mod.modifyKeyDeserializer(config, type, deser);
        }
    }
    return deser;
}