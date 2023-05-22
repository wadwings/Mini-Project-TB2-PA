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

    class distanceMethodType:
        tcrdist = 'tcrdist'
        gliph = 'gliph'
        giana = 'giana'
        tcrpeg = 'tcrpeg'

    class feMethodType:
        giana_features = 'giana_features'
        tcrpeg = 'tcrpeg'
        brute_force = 'brute_force'
        distance_metrics = 'distance_metrics'

    class clusterMethodType:
        KMeans = 'kmeans'

    # dimension_reduction_method_type
    class drMethodType:
        pca = 'pca'

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
        distanceMethodType.tcrdist: {'cdr3.alpha': 'cdr3_a_aa', "v.alpha": "v_a_gene", "j.alpha": "j_a_gene",
                                     'cdr3.beta': 'cdr3_b_aa', 'v.beta': 'v_b_gene', 'j.beta': 'j_b_gene'},
        distanceMethodType.gliph: {'cdr3.alpha': 'CDR3a', "v.alpha": "TRAV", "j.alpha": "TRAJ",
                                   'cdr3.beta': 'CDR3b', 'v.beta': 'TRBV', 'j.beta': 'TRBJ'},
        distanceMethodType.giana: {'cdr3.beta': 'aminoAcid', 'v.beta': 'vMaxResolved'},
        distanceMethodType.tcrpeg: {'cdr3.beta': 'aa', 'v.beta': 'v', 'j.beta': 'j'}
    }

    columns = {
        chainType.alpha + distanceMethodType.tcrdist: ['cdr3_a_aa', 'v_a_gene', "j_a_gene"],
        chainType.beta + distanceMethodType.tcrdist: ['cdr3_b_aa', 'v_b_gene', "j_b_gene"],
        chainType.pw_ab + distanceMethodType.tcrdist: ['cdr3_a_aa', 'v_a_gene', "j_a_gene", 'cdr3_b_aa', 'v_b_gene',
                                                       "j_b_gene"],
        chainType.alpha + distanceMethodType.gliph: ['CDR3a', 'TRAV', 'TRAJ'],
        chainType.beta + distanceMethodType.gliph: ['CDR3b', 'TRBV', 'TRBJ'],
        chainType.pw_ab + distanceMethodType.gliph: ['CDR3a', 'TRAV', 'TRAJ', 'CDR3b', 'TRBV', 'TRBJ'],
        chainType.alpha + distanceMethodType.giana: ['aminoAcid', 'vMaxResolved'],
        chainType.beta + distanceMethodType.giana: ['aminoAcid', 'vMaxResolved'],
        chainType.pw_ab + distanceMethodType.giana: ['aminoAcid', 'vMaxResolved'],
        chainType.beta + distanceMethodType.tcrpeg: ['aa']
    }

    gene_columns = {
        chainType.alpha: ['cdr3_a_aa'],
        chainType.beta: ['cdr3_b_aa'],
        chainType.pw_ab: ['cdr3_a_aa', 'cdr3_b_aa'],
    }

    species: speciesType.human
    chain: chainType.alpha
    label: labelType.none
    distance_method: distanceMethodType.tcrdist
    fe_method: feMethodType.giana_features
    cluster_method: clusterMethodType.KMeans
    dr_method: drMethodType.pca

    def __init__(self, species=speciesType.human, chain=chainType.alpha, label=labelType.none,
                 distance_method=distanceMethodType.tcrdist, feature_extract_method=feMethodType.giana_features,
                 cluster_method=clusterMethodType.KMeans, dr_method=drMethodType.pca):
        self.species = species
        self.chain = chain
        self.label = label
        self.distance_method = distance_method
        self.fe_method = feature_extract_method
        self.cluster_method = cluster_method
        self.dr_method = dr_method

    def set_config(self, species, chain):
        self.species = species
        self.chain = chain

    def set_chain(self, chain):
        self.chain = chain

    def set_species(self, species):
        self.species = species

    def set_label(self, label):
        self.label = label

    def set_distance_method(self, method):
        self.distance_method = method
        # giana and gliph can only be used with beta chain
        if (
                method == self.distanceMethodType.giana or
                method == self.distanceMethodType.gliph) and self.chain != self.chainType.beta:
            print(f"{method} cannot be used within self.chain = {self.chain}")

    def set_fe_method(self, method):
        self.fe_method = method

    def set_dr_method(self, method):
        self.dr_method = method

    def set_clustering_method(self, method):
        self.cluster_method = method

    def get_label_columns(self):
        return copy.deepcopy(self.labelColumns[self.label])

    def get_species(self):
        return copy.deepcopy(self.speciesString[self.species])

    def get_chain(self):
        return copy.deepcopy(self.chainsList[self.chain])

    def get_distance_method(self):
        return copy.deepcopy(self.distance_method)

    def get_fe_method(self):
        return copy.deepcopy(self.fe_method)

    def get_clustering_method(self):
        return copy.deepcopy(self.cluster_method)

    def get_dr_method(self):
        return copy.deepcopy(self.dr_method)

    def get_columns(self):
        return copy.deepcopy(self.columns[self.chain + self.distance_method])

    def get_columns_mapping(self):
        return copy.deepcopy(self.mapping_columns[self.distance_method])

    def get_gene_columns(self):
        return copy.deepcopy(self.gene_columns[self.chain])

    def get_species_name(self):
        return copy.deepcopy(self.species)

    def clear(self):
        self.species = self.speciesType.human
        self.chain = self.chainType.alpha
        self.label = self.labelType.none
        self.distance_method = self.distanceMethodType.tcrdist
        self.fe_method = self.feMethodType.giana_features
        self.cluster_method = self.clusterMethodType.KMeans
        self.dr_method = self.drMethodType.pca


config = Config()

if __name__ == '__main__':
    config = Config()
