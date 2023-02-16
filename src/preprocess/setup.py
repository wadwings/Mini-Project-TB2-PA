class Config:
    class speciesType:
        human = 'HomoSapiens'
        mouse = 'MusMusculus'
        monkey = 'MacacaMulatta'

    class chainType:
        alpha = 'alpha'
        beta = 'beta'
        pw_ab = 'pw_ab'

    class labelType:
        mhc_a = 'mhc.a'
        mhc_b = 'mhc.b'
        mhc_class = 'mhc.class'
        mhc = 'mhc'
        epitope = 'antigen.epitope'
        gene = 'antigen.gene'
        species = 'antigen.species'
        antigen = 'antigen'
        mhc_antigen = 'mhc_antigen'

    labelColumns = {
        labelType.mhc_a: ['mhc.a'],
        labelType.mhc_b: ['mhc.b'],
        labelType.mhc_class: ['mhc.class'],
        labelType.mhc: ['mhc.a', 'mhc.b', 'mhc.class'],
        labelType.epitope: ['antigen.epitope'],
        labelType.gene: ['antigen.gene'],
        labelType.species: ['antigen.species'],
        labelType.antigen: ['antigen.epitope', 'antigen.gene', 'antigen.species'],
        labelType.mhc_antigen: ['mhc.a', 'mhc.b', 'mhc.class', 'antigen.epitope', 'antigen.gene', 'antigen.species'],
    }

    chainsList = {
        chainType.alpha: ['alpha'],
        chainType.beta: ['beta'],
        chainType.pw_ab: ['alpha', 'beta'],
    }

    speciesString = {
        speciesType.human: 'human',
        speciesType.mouse: 'mouse',
        speciesType.monkey: 'monkey'
    }

    columns = {
        chainType.alpha: ['cdr3_a_aa', 'v_a_gene', "j_a_gene"],
        chainType.beta: ['cdr3_b_aa', 'v_b_gene', "j_b_gene"],
        chainType.pw_ab: ['cdr3_a_aa', 'v_a_gene', "j_a_gene", 'cdr3_b_aa', 'v_b_gene', "j_b_gene"],
    }

    gene_columns = {
        chainType.alpha: ['cdr3_a_aa'],
        chainType.beta: ['cdr3_b_aa'],
        chainType.pw_ab: ['cdr3_a_aa', 'cdr3_b_aa'],
    }

    species: speciesType.human
    chain: chainType.alpha
    label: labelType.mhc

    def __init__(self, species=speciesType.human, chain=chainType.alpha, label=labelType.mhc):
        self.species = species
        self.chain = chain
        self.label = label

    def set_config(self, species, chain):
        self.species = species
        self.chain = chain

    def set_chain(self, chain):
        self.chain = chain

    def set_species(self, species):
        self.species = species

    def set_label(self, label):
        self.label = label

    def get_label_columns(self):
        return self.labelColumns[self.label]

    def get_species(self):
        return self.speciesString[self.species]

    def get_chain(self):
        return self.chainsList[self.chain]

    def get_columns(self):
        return self.columns[self.chain]

    def get_gene_columns(self):
        return self.gene_columns[self.chain]

    def get_species_name(self):
        return self.species


config = Config()

if __name__ == '__main__':
    config = Config()
