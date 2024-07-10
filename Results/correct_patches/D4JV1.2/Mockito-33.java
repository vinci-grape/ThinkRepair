public boolean hasSameMethod(Invocation candidate) {        
    //not using method.equals() for 1 good reason:
    //sometimes java generates forwarding methods when generics are in play see JavaGenericsForwardingMethodsTest
    Method m1 = invocation.getMethod();
    Method m2 = candidate.getMethod();

    if (m1.getName().equals(m2.getName()) && m1.getParameterTypes().length == m2.getParameterTypes().length) {
        for (int i = 0; i < m1.getParameterTypes().length; i++) {
            if (!m1.getParameterTypes()[i].equals(m2.getParameterTypes()[i])) {
                return false;
            }
        }
        return true;
    }
    return false;
}