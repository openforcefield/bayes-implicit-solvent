import numpy as np
from bayes_implicit_solvent.typers import FlatGBTyper

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
            new_typing_scheme = FlatGBTyper(new_smarts_list)
            new_gb_model = GBModel(new_typing_scheme, new_radii)

            log_p_forward = np.log(prob_add / n_available_primitives)
            log_p_reverse = np.log(prob_reverse_delete / len(self.primitives))
            log_p_forward_over_reverse = log_p_forward - log_p_reverse

        else:  # proposing a deletion
            new_typing_scheme = FlatGBTyper(initial_model.typing_scheme.smarts_list[:-1])
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

        if len(initial_model.typing_scheme.smarts_list) == 1: # don't touch 0th element, assume it's a wildcard
            prob_add = 1.0
            insertion_ind = 1

        else:
            prob_add = default_prob_add
            insertion_ind = np.random.randint(1, len(initial_model.radii) + 1)

        cloning_ind = insertion_ind - 1
        available_primitives = list(self.primitives)
         # don't touch 0th element, assume it's a wildcard
        available_primitives.remove(initial_model.typing_scheme.smarts_list[cloning_ind])
        n_available_primitives = len(available_primitives)

        # TODO: Double-check this logic for length-1 lists...

        add = np.random.rand() < prob_add



        if add:  # propose an addition
            new_primitive = available_primitives[np.random.randint(n_available_primitives)]

            new_radii = list(initial_model.radii)
            new_radii.insert(insertion_ind, new_radii[cloning_ind])
            new_radii = np.array(new_radii)

            new_smarts_list = list(initial_model.typing_scheme.smarts_list)
            new_smarts_list.insert(insertion_ind, new_primitive)

            new_typing_scheme = FlatGBTyper(new_smarts_list)
            new_gb_model = GBModel(new_typing_scheme, new_radii)

            log_p_forward = np.log(prob_add / (n_available_primitives * len(initial_model.radii - 1)))
            log_p_reverse = np.log(prob_reverse_delete / (len(self.primitives) * len(new_radii)))

        else:  # proposing a deletion
            new_radii = list(initial_model.radii)
            ind_to_remove = np.random.randint(1, len(new_radii))
            _ = new_radii.pop(ind_to_remove)
            new_radii = np.array(new_radii)

            new_smarts_list = list(initial_model.typing_scheme.smarts_list)
            _ = new_smarts_list.pop(ind_to_remove)

            new_typing_scheme = FlatGBTyper(new_smarts_list)
            new_gb_model = GBModel(new_typing_scheme, new_radii)

            log_p_forward = np.log(prob_reverse_delete / (len(self.primitives) * len(initial_model.radii - 1)))
            log_p_reverse = np.log((1 - prob_reverse_delete) / (len(self.primitives) * len(new_radii)))

        log_p_forward_over_reverse = log_p_forward - log_p_reverse

        return {'proposal': new_gb_model, 'log_p_forward_over_reverse': log_p_forward_over_reverse}


class AddOrDeleteElaboration(Proposal):
    # selects a position in the list uniformly at random, and makes a "child" type that decorates / elaborates on the parent type

    def __init__(self):
        raise(NotImplementedError())

###### Merge-split ######
class MergeSplitDisjunction(Proposal):
    # merge: select two consecutive positions in the list randomly,
    # and create a new smarts pattern that "ORs" them together.
    # set the new radius to be the average of the two original radii.
    # remove the original two types.

    # split: select a random position, and if it contains a disjunction, create
    # two new types, one for each term in the disjunction. (If it contains more than one disjunction, pick one of them at random to split on.)
    # if it doesn't contain a disjunction, then duplicate?
    # set both new radii equal to the previous radius

    def __init__(self):
        raise(NotImplementedError())

class MergeSplitConjunction(Proposal):
    # merge: select two (potentially nonunique) positions in the list randomly,
    # and create a new smarts pattern that "ANDs" them together.
    # set the new

    # split: select a random position, and if it contains a conjunction, create
    # two new types, one for each term in the conjunction. (If it contains more than one conjunction, pick one of them at random to split on.)
    # if it doesn't contain a conjunction, then duplicate?
    # set both new radii equal to the previous radius


    def __init__(self):
        raise(NotImplementedError())


class SwapTwoPatterns(Proposal):
    """select two indices and swap them, avoiding touching the 0th element of the list"""

    def __init__(self):
        pass

    def sample_proposal(self, initial_model):
        old_smarts_list = initial_model.typing_scheme.smarts_list
        old_radii = initial_model.radii

        if len(initial_model.typing_scheme.smarts_list) <= 2:
            new_gb_model = GBModel(FlatGBTyper(old_smarts_list), old_radii)
            return {'proposal': new_gb_model, 'log_p_forward_over_reverse': 0}
        else:
            indices_to_swap = np.random.randint(1, len(initial_model.typing_scheme.smarts_list), 2)
            new_radii = np.array(old_radii)
            new_smarts_list = list(old_smarts_list)

            new_radii[indices_to_swap] = new_radii[indices_to_swap[::-1]]
            first_smarts = new_smarts_list[indices_to_swap[0]]
            new_smarts_list[indices_to_swap[0]] = new_smarts_list[indices_to_swap[1]]
            new_smarts_list[indices_to_swap[1]] = first_smarts

            new_gb_model = GBModel(FlatGBTyper(new_smarts_list), new_radii)

            return {'proposal': new_gb_model, 'log_p_forward_over_reverse': 0}

class MultiProposal(Proposal):
    """randomly select from a list of potential proposals"""

    def __init__(self, proposals, weights=None):
        self.proposals = proposals

        if type(weights) == type(None):
            self.weights = np.ones(len(proposals))
        else:
            self.weights = np.array(weights)
        self.weights /= np.sum(self.weights)

    def sample_proposal(self, initial_model):
        proposal_object = self.proposals[np.random.choice(len(self.weights), p=self.weights)]
        print('sampling cross-model proposal: ', proposal_object.__class__.__name__)
        return proposal_object.sample_proposal(initial_model)
