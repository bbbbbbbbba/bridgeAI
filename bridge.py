import random

# TODO: trump contract
class BridgePosition:
    def __init__(self, hands = None):
        if hands is None:
            deck = [(suit, rank) for suit in range(4) for rank in range(2, 15)]
            random.shuffle(deck)
            self.hands = []
            for i in range(4):
                hand = [[] for suit in range(4)]
                for j in range(13):
                    suit, rank = deck.pop()
                    hand[suit].append(rank)
                for suit in range(4): hand[suit].sort(reverse = True)
                self.hands.append(hand)
        else:
            self.hands = hands
        self.currentPlayer = 0
        self.cardsInTrick = 0
        self.currentSuit = None
        self.currentHigh = None
        self.currentHighPlayer = None

    def getMoves(self):
        if self.cardsInTrick == 0 or self.hands[self.currentPlayer][self.currentSuit] == []: suits = range(4)
        else: suits = [self.currentSuit]
        res = []
        for suit in suits:
            for rank in self.hands[self.currentPlayer][suit]:
                res.append((suit, rank))
        return res

    def makeMove(self, move):
        suit, rank = move
        self.hands[self.currentPlayer][suit].remove(rank)
        if self.cardsInTrick == 0: self.currentSuit = suit
        if self.cardsInTrick == 0 or suit == self.currentSuit and rank > self.currentHigh:
            self.currentHigh = rank
            self.currentHighPlayer = self.currentPlayer
        if self.cardsInTrick == 3:
            trickWon = (self.currentPlayer - self.currentHighPlayer) % 2 == 0
            self.currentPlayer = self.currentHighPlayer
            self.cardsInTrick = 0
            self.currentSuit = None
            self.currentHigh = None
            self.currentHighPlayer = None
            return trickWon
        self.currentPlayer = (self.currentPlayer + 1) % 4
        self.cardsInTrick += 1
        return False

    # Converts the position to a tensor for use with TensorFlow (currently a vector)
    def toTensor(self):
        res = []
        for i in range(4):
            hand = self.hands[(self.currentPlayer + i) % 4]
            for suit in range(4):
                for rank in range(2, 15):
                    res.append(1 if rank in hand[suit] else 0)
        for i in range(4): res.append(1 if self.cardsInTrick == i else 0)
        for suit in range(4): res.append(1 if self.currentSuit == suit else 0)
        for rank in range(2, 15): res.append(1 if self.currentHigh == rank else 0)
        for i in range(1, 4): res.append(1 if self.currentHighPlayer == (self.currentPlayer + i) % 4 else 0)
        return res
