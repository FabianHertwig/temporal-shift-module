

class Gestures:
    def __init__(self):
        self.gestures = [
            "Doing other things",  # 0
            "Drumming Fingers",  # 1
            "No gesture",  # 2
            "Pulling Hand In",  # 3
            "Pulling Two Fingers In",  # 4
            "Pushing Hand Away",  # 5
            "Pushing Two Fingers Away",  # 6
            "Rolling Hand Backward",  # 7
            "Rolling Hand Forward",  # 8
            "Shaking Hand",  # 9
            "Sliding Two Fingers Down",  # 10
            "Sliding Two Fingers Left",  # 11
            "Sliding Two Fingers Right",  # 12
            "Sliding Two Fingers Up",  # 13
            "Stop Sign",  # 14
            "Swiping Down",  # 15
            "Swiping Left",  # 16
            "Swiping Right",  # 17
            "Swiping Up",  # 18
            "Thumb Down",  # 19
            "Thumb Up",  # 20
            "Turning Hand Clockwise",  # 21
            "Turning Hand Counterclockwise",  # 22
            "Zooming In With Full Hand",  # 23
            "Zooming In With Two Fingers",  # 24
            "Zooming Out With Full Hand",  # 25
            "Zooming Out With Two Fingers"  # 26
        ]

    def get_name(self, id: int):
        return self.gestures[id]