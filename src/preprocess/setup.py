import copy

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
        none = ''
        mhc_a = 'mhc.a'
        mhc_b = 'mhc.b'
        mhc_class = 'mhc.class'
        mhc = 'mhc'
        epitope = 'antigen.epitope'
        gene = 'antigen.gene'
        species = 'antigen.species'
        antigen = 'antigen'
        mhc_antigen = 'mhc_antigen'

    class methodType:
        tcrdist = 'tcrdist'
        gliph = 'gliph'
        giana = 'giana'

    labelColumns = {
        labelType.none: [],
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

    mapping_columns = {
        methodType.tcrdist: {'cdr3.alpha': 'cdr3_a_aa', "v.alpha": "v_a_gene", "j.alpha": "j_a_gene",
                             'cdr3.beta': 'cdr3_b_aa', 'v.beta': 'v_b_gene', 'j.beta': 'j_b_gene'},
        methodType.gliph: {'cdr3.alpha': 'CDR3a', "v.alpha": "TRAV", "j.alpha": "TRAJ",
                           'cdr3.beta': 'CDR3b', 'v.beta': 'TRBV', 'j.beta': 'TRBJ'},
        methodType.giana: {'cdr3.beta': 'aminoAcid', 'v.beta': 'vMaxResolved'},
    }

    columns = {
        chainType.alpha + methodType.tcrdist: ['cdr3_a_aa', 'v_a_gene', "j_a_gene"],
        chainType.beta + methodType.tcrdist: ['cdr3_b_aa', 'v_b_gene', "j_b_gene"],
        chainType.pw_ab + methodType.tcrdist: ['cdr3_a_aa', 'v_a_gene', "j_a_gene", 'cdr3_b_aa', 'v_b_gene',
                                               "j_b_gene"],
        chainType.alpha + methodType.gliph: ['CDR3a', 'TRAV', 'TRAJ'],
        chainType.beta + methodType.gliph: ['CDR3b', 'TRBV', 'TRBJ'],
        chainType.pw_ab + methodType.gliph: ['CDR3a', 'TRAV', 'TRAJ', 'CDR3b', 'TRBV', 'TRBJ'],
        chainType.alpha + methodType.giana: ['aminoAcid', 'vMaxResolved'],
        chainType.beta + methodType.giana: ['aminoAcid', 'vMaxResolved'],
        chainType.pw_ab + methodType.giana: ['aminoAcid', 'vMaxResolved'],
    }

    gene_columns = {
        chainType.alpha: ['cdr3_a_aa'],
        chainType.beta: ['cdr3_b_aa'],
        chainType.pw_ab: ['cdr3_a_aa', 'cdr3_b_aa'],
    }

    species: speciesType.human
    chain: chainType.alpha
    label: labelType.none
    method: methodType.tcrdist

    def __init__(self, species=speciesType.human, chain=chainType.alpha, label=labelType.none,
                 method=methodType.tcrdist):
        self.species = species
        self.chain = chain
        self.label = label
        self.method = method

    def set_config(self, species, chain):
        self.species = species
        self.chain = chain

    def set_chain(self, chain):
        self.chain = chain

    def set_species(self, species):
        self.species = species

    def set_label(self, label):
        self.label = label

    def set_method(self, method):
        self.method = method
        if (method == self.methodType.giana or method == self.methodType.gliph) and self.chain != self.chainType.beta:
            print(f"{method} cannot be used within self.chain = {self.chain}")

    def get_label_columns(self):
        return copy.deepcopy(self.labelColumns[self.label])

    def get_species(self):
        return copy.deepcopy(self.speciesString[self.species])

    def get_chain(self):
        return copy.deepcopy(self.chainsList[self.chain])

    def get_method(self):
        return copy.deepcopy(self.method)

    def get_columns(self):
        return copy.deepcopy(self.columns[self.chain + self.method])

    def get_columns_mapping(self):
        return copy.deepcopy(self.mapping_columns[self.method])

    def get_gene_columns(self):
        return copy.deepcopy(self.gene_columns[self.chain])

    def get_species_name(self):
        return copy.deepcopy(self.species)


config = Config()

if __name__ == '__main__':
    config = Config()
