// Fixed Function
public static Number createNumber(final String str) {
    if (str == null) {
        return null;
    }
    if (StringUtils.isBlank(str)) {
        throw new NumberFormatException("A blank string is not a valid number");
    }
    // Need to deal with all possible hex prefixes here
    final String[] hexPrefixes = {"0x", "0X", "#"};
    final int length = str.length();
    final int offset = str.charAt(0) == '+' || str.charAt(0) == '-' ? 1 : 0;
    int pfxLen = 0;
    for (final String pfx : hexPrefixes) {
        if (str.startsWith(pfx, offset)) {
            pfxLen += pfx.length() + offset;
            break;
        }
    }
    if (pfxLen > 0) { // we have a hex number
        char firstSigDigit = 0; // strip leading zeroes
        for (int i = pfxLen; i < length; i++) {
            firstSigDigit = str.charAt(i);
            if (firstSigDigit != '0') {
                break;
            }
            pfxLen++;
        }
        final int hexDigits = length - pfxLen;
        if (hexDigits > 16 || hexDigits == 16 && firstSigDigit > '7') { // too many for Long
            return createBigInteger(str);
        }
        if (hexDigits > 8 || hexDigits == 8 && firstSigDigit > '7') { // too many for an int
            return createLong(str);
        }
        return createInteger(str);
    }
    final char lastChar = str.charAt(length - 1);
    final String mant;
    final String dec;
    final String exp;
    final int decPos = str.indexOf('.');
    final int expPos = str.indexOf('e', offset + 1) != -1 || str.indexOf('E', offset + 1) != -1 ?
            Math.max(str.indexOf('e', offset + 1), str.indexOf('E', offset + 1)) : -1;

    // Detect if the return type has been requested
    final boolean requestType = !Character.isDigit(lastChar) && lastChar != '.';
    if (decPos > -1) { // there is a decimal point
        if (expPos > -1) { // there is an exponent
            if (expPos < decPos || expPos > length) {
                throw new NumberFormatException(str + " is not a valid number.");
            }
            dec = str.substring(decPos + 1, expPos);
        } else {
            // No exponent, but there may be a type character to remove
            dec = str.substring(decPos + 1, requestType ? length - 1 : length);
        }
        mant = getMantissa(str, decPos);
    } else {
        if (expPos > -1) {
            if (expPos > length) {
                throw new NumberFormatException(str + " is not a valid number.");
            }
            mant = getMantissa(str, expPos);
        } else {
            // No decimal, no exponent, but there may be a type character to remove
            mant = getMantissa(str, requestType ? length - 1 : length);
        }
        dec = null;
    }

    if (requestType) {
        if (expPos > -1 && expPos < length - 1) {
            exp = str.substring(expPos + 1, length - 1);
        } else {
            exp = null;
        }
        final String numeric = str.substring(0, length - 1);
        switch (lastChar) {
            case 'l' :
            case 'L' :
                if (dec == null && exp == null &&
                        (!numeric.isEmpty() && numeric.charAt(0) == '-' && isDigits(numeric.substring(1)) || isDigits(numeric))) {
                    try {
                        return createLong(numeric);
                    } catch (final NumberFormatException ignored) {
                        // Too big for a long
                    }
                    return createBigInteger(numeric);
                }
                throw new NumberFormatException(str + " is not a valid number.");
            case 'f' :
            case 'F' :
                try {
                    final Float f = createFloat(str);
                    if (!(f.isInfinite() || f.floatValue() == 0.0F && !isZero(mant, dec))) {
                        return f;
                    }
                } catch (final NumberFormatException ignored) {
                    // ignore the bad number
                }
                //$FALL-THROUGH$
            case 'd' :
            case 'D' :
                try {
                    final Double d = createDouble(str);
                    if (!(d.isInfinite() || d.doubleValue() == 0.0D && !isZero(mant, dec))) {
                        return d;
                    }
                } catch (final NumberFormatException ignored) {
                    // ignore the bad number
                }
                try {
                    return createBigDecimal(numeric);
                } catch (final NumberFormatException ignored) {
                    // ignore the bad number
                }
                //$FALL-THROUGH$
            default :
                throw new NumberFormatException(str + " is not a valid number.");
        }
    }

    if (expPos > -1 && expPos < length - 1) {
        exp = str.substring(expPos + 1);
    } else {
        exp = null;
    }
    if (dec == null && exp == null) {
        try {
            return createInteger(str);
        } catch (final NumberFormatException ignored) {
            // ignore the bad number
        }
        try {
            return createLong(str);
        } catch (final NumberFormatException ignored) {
            // ignore the bad number
        }
        return createBigInteger(str);
    }

    try {
        final Float f = createFloat(str);
        final Double d = createDouble(str);
        if (!f.isInfinite() && !(f.floatValue() == 0.0F && !isZero(mant, dec)) &&
                f.toString().equals(d.toString())) {
            return f;
        }
        if (!d.isInfinite() && !(d.doubleValue() == 0.0D && !isZero(mant, dec))) {
            final BigDecimal b = createBigDecimal(str);
            if (b.compareTo(BigDecimal.valueOf(d.doubleValue())) == 0) {
                return d;
            }
            return b;
        }
    } catch (final NumberFormatException ignored) {
        // ignore the bad number
    }
    return createBigDecimal(str);
}