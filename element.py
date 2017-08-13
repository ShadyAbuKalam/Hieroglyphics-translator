class Element:
    def __init__(self, image, location):
        self.image = image
        self.location = location  # Actual location in original image

    def __getattribute__(self, name: str):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            pass
        if name == 'x':
            return self.location[0]
        elif name == 'y':
            return self.location[1]
        elif name == 'width':
            return self.image.shape[1]
        elif name == 'height':
            return self.image.shape[0]
        else:
            return self.image.__getattribute__(name)
