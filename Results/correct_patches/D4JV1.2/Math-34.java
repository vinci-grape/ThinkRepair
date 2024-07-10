public Iterator<Chromosome> iterator() {
    return Collections.unmodifiableList(chromosomes).iterator();
}