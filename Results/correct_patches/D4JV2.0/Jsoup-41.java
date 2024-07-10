public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    if (!super.equals(o)) return false;

    Element element = (Element) o;

    if (!this.tag().equals(element.tag())) { // compare tags as well
        return false;
    }

    return Objects.equals(this.attributes(), element.attributes()); // compare attribute maps
}