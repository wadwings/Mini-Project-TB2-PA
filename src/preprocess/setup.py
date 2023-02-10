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

        beta = 'beta'
        pw_ab = 'pw_ab'

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

    species: speciesType.human
    chain: chainType.alpha
    label: labelType.None

    def setConfig(self, species, chain):
        self.species = species
        self.chain = chain

    def setChain(self, chain):
        self.chain = chain

    def setSpecies(self, species):
        self.species = species

    def getSpecies(self):
        return self.speciesString[self.species]

    def getChain(self):
        return self.chainsList[self.chain]

    def getColumns(self):
        return self.columns[self.chain]

    def getSpeciesName(self):
        return self.species


config = Config()

if __name__ == '__main__':
    config = Config()
