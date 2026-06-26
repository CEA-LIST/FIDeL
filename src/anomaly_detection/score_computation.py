def compute_score(self, output, input):
    if self.ad != "Representation":
        loss = self.criterion(output, input)
        score = loss.item()

    elif self.ad == "Representation":
        score = output

    else:
        raise NotImplementedError

    return score
