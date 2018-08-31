import numpy as np
from bayes_implicit_solvent.typers import GBTyper

class Proposal():
    """abstract class for a proposal distribution:
    must support sample_proposal, may optionally support
    evaluating the proposal log probability on arbitrary pairs of states"""

    def sample_proposal(self, initial):
        raise (NotImplementedError("Must return a dictionary with keys 'proposal' and 'log_p_forward_over_reverse'"))

    def evaluate_proposal_log_probability(self, initial, final):
        raise (NotImplementedError())

from collections import namedtuple

GBModel = namedtuple('GBModel', ['typing_scheme', 'radii'])

class RadiusInheritanceProposal(Proposal):
    # forward proposal samples p(radius_new | radius_old)
    # some options include:
    # * p(radius_new | radius_old) = elta(radius_new - radius_old)
    # * p(radius_new | radius_old) = prior(radius_new)
    # * p(radius_new | radius_old) = Normal(radius_new | mean=radius_old, sigma=sigma_proposal)

    # we might generalize this to p(radius_new | radii_array) or something

    def __init__(self):
        raise(NotImplementedError())


##### Birth-death #####

class AddOrDeletePrimitiveAtEndOfList(Proposal):
    def __init__(self, primitives):
        """Sample a SMARTS pattern uniformly from a list of primitives,
        and append a new type to the end of the list with this SMARTS pattern
        and the radius of the last """
        self.primitives = primitives
        self.n_primitives = len(primitives)

    def sample_proposal(self, initial_model):

        if initial_model.typing_scheme.smarts_list[-1] not in self.primitives:
            raise (RuntimeError(
                "Couldn't apply this move because the last element of initial_model.smarts_list wasn't a primitive. Ratio of forward and reverse proposal probabilities not finite..."))

        default_prob_add = 0.5  # default
        prob_reverse_delete = 1.0 - default_prob_add

        if len(initial_model.typing_scheme.smarts_list) == 1:
            prob_add = 1.0
        else:
            prob_add = default_prob_add

        # TODO: Double-check this logic for length-1 lists...

        add = np.random.rand() < prob_add

        available_primitives = list(self.primitives)
        available_primitives.remove(initial_model.typing_scheme.smarts_list[-1])
        n_available_primitives = len(available_primitives)
        # don't propose to add a consecutive duplicate
        # (non-consecutive duplicates are fine, because they can mean different things due to last-match-wins behavior)

        if add:  # propose an addition
            new_primitive = available_primitives[np.random.randint(n_available_primitives)]
            new_radii = np.zeros(len(initial_model.radii) + 1)
            new_radii[:-1] = initial_model.radii
            new_radii[-1] = initial_model.radii[-1]
            new_smarts_list = initial_model.typing_scheme.smarts_list + [new_primitive]
            new_typing_scheme = GBTyper(new_smarts_list)
            new_gb_model = GBModel(new_typing_scheme, new_radii)

            log_p_forward = np.log(prob_add / n_available_primitives)
            log_p_reverse = np.log(prob_reverse_delete / len(self.primitives))
            log_p_forward_over_reverse = log_p_forward - log_p_reverse

        else:  # proposing a deletion
            new_typing_scheme = GBTyper(initial_model.typing_scheme.smarts_list[:-1])
            new_radii = initial_model.radii[:-1]
            new_gb_model = GBModel(new_typing_scheme, new_radii)

            log_p_forward = np.log(prob_reverse_delete / len(self.primitives))
            log_p_reverse = np.log((1 - prob_reverse_delete) / n_available_primitives)
            log_p_forward_over_reverse = log_p_forward - log_p_reverse

        return {'proposal': new_gb_model, 'log_p_forward_over_reverse': log_p_forward_over_reverse}

class AddOrDeletePrimitiveAtRandomPositionInList(Proposal):
    # selects a position in the list uniformly at random, excluding the beginning of the list, and adds a new primitive there with a
    # TODO: add the two optimizations from the above proposal (don't touch wildcard, don't introduce duplicates)

    def __init__(self, primitives):
        """Sample a SMARTS pattern uniformly from a list of primitives,
        and insert a new type with this SMARTS pattern
        and the radius of the one immediately preceding it """
        self.primitives = primitives
        self.n_primitives = len(primitives)

    def sample_proposal(self, initial_model):


        if not np.all([pattern in self.primitives for pattern in initial_model.typing_scheme.smarts_list]):
            raise (RuntimeError(
                "Couldn't apply this move because initial_model.smarts_list contains non-primitives. Ratio of forward and reverse proposal probabilities not finite..."))

        default_prob_add = 0.5  # default
        prob_reverse_delete = 1.0 - default_prob_add

        if len(initial_model.typing_scheme.smarts_list) <= 2: # don't touch 0th element, assume it's a wildcard
            prob_add = 1.0
        else:
            prob_add = default_prob_add

        # TODO: Double-check this logic for length-1 lists...

        add = np.random.rand() < prob_add
        ind = np.random.randint(1, len(initial_model.radii)) # don't touch 0th element, assume it's a wildcard

        if add:  # propose an addition
            new_primitive = self.primitives[np.random.randint(self.n_primitives)]

            new_radii = list(initial_model.radii)
            new_radii.insert(ind, new_radii[ind])
            new_radii = np.array(new_radii)

            new_smarts_list = list(initial_model.typing_scheme.smarts_list)
            new_smarts_list.insert(ind, new_primitive)

            new_typing_scheme = GBTyper(new_smarts_list)
            new_gb_model = GBModel(new_typing_scheme, new_radii)

            log_p_forward = np.log(prob_add / (len(self.primitives) * len(initial_model.radii)))
            log_p_reverse = np.log(prob_reverse_delete / (len(self.primitives) * len(new_radii)))

        else:  # proposing a deletion

            new_radii = list(initial_model.radii)
            _ = new_radii.pop(ind)
            new_radii = np.array(new_radii)

            new_smarts_list = list(initial_model.typing_scheme.smarts_list)
            _ = new_smarts_list.pop(ind)

            new_typing_scheme = GBTyper(new_smarts_list)
            new_gb_model = GBModel(new_typing_scheme, new_radii)

            log_p_forward = np.log(prob_reverse_delete / (len(self.primitives) * len(initial_model.radii)))
            log_p_reverse = np.log((1 - prob_reverse_delete) / (len(self.primitives) * len(new_radii)))

        log_p_forward_over_reverse = log_p_forward - log_p_reverse

        return {'proposal': new_gb_model, 'log_p_forward_over_reverse': log_p_forward_over_reverse}


class AddOrDeleteElaboration(Proposal):
    # selects a position in the list uniformly at random, and makes a "child" type that decorates / elaborates on the parent type

    def __init__(self):
        raise(NotImplementedError())

###### Merge-split ######
class MergeSplitDisjunction(Proposal):
    # merge: select two (potentially nonunique) positions in the list randomly,
    # and create a new smarts pattern that contains "ORs" them together

    def __init__(self):
        raise(NotImplementedError())

class MergeSplitConjunctionjunction(Proposal):
    # merge: select two (potentially nonunique) positions in the list randomly,
    # and create a new smarts pattern that contains "ANDs" them together

    def __init__(self):
        raise(NotImplementedError())


