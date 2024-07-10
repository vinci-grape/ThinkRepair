protected Object deserializeUsingPropertyBasedWithUnwrapped(JsonParser p,
		DeserializationContext ctxt)
    throws IOException, JsonProcessingException
{
    final PropertyBasedCreator creator = _propertyBasedCreator;
    PropertyValueBuffer buffer = creator.startBuilding(p, ctxt, _objectIdReader);

    TokenBuffer tokens = new TokenBuffer(p, ctxt);
    tokens.writeStartObject();

    JsonToken t = p.getCurrentToken();
    while (t == JsonToken.FIELD_NAME) {
        String propName = p.getCurrentName();
        p.nextToken(); // Move to value
        // Creator property?
        SettableBeanProperty creatorProp = creator.findCreatorProperty(propName);
        if (creatorProp != null) {
            // Assign parameter using creator property
            buffer.assignParameter(creatorProp, creatorProp.deserialize(p, ctxt));
            t = p.nextToken(); // Move to next field
            continue;
        }
        // Object Id property?
        if (buffer.readIdProperty(propName)) {
            t = p.nextToken(); // Move to next field
            continue;
        }
        // Regular property? Buffer it
        SettableBeanProperty prop = _beanProperties.find(propName);
        if (prop != null) {
            buffer.bufferProperty(prop, prop.deserialize(p, ctxt));
            t = p.nextToken(); // Move to next field
            continue;
        }
        if (_ignorableProps != null && _ignorableProps.contains(propName)) {
            handleIgnoredProperty(p, ctxt, handledType(), propName);
            t = p.nextToken(); // Move to next field
            continue;
        }
        tokens.writeFieldName(propName);
        tokens.copyCurrentStructure(p);
        // "Any property"?
        if (_anySetter != null) {
            buffer.bufferAnyProperty(_anySetter, propName, _anySetter.deserialize(p, ctxt));
        }
        t = p.nextToken(); // Move to next field
    }

    tokens.writeEndObject();
    // Build object using creator inside the loop, not outside
    Object bean;
    try {
        bean = creator.build(ctxt, buffer);
    } catch (Exception e) {
        return wrapInstantiationProblem(e, ctxt);
    }
    // Process unwrapped properties
    return _unwrappedPropertyHandler.processUnwrapped(p, ctxt, bean, tokens);
}