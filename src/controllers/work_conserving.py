class WorkConserving:
    def action(self, section):
        masks = section.masks()
        return 1 if masks[1] else 2 if masks[2] else 0
