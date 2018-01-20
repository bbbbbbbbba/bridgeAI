import random

suit_str = 'SHDC'
rank_str = '  23456789TJQKA'

def move_to_str(move):
    return '%s%s' % (suit_str[move[0]], rank_str[move[1]])

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
        self.score = 0 # Number of tricks won by players 0 and 2
        self.move_to_str = move_to_str

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
            if self.currentHighPlayer % 2 == 0: self.score += 1
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

    def visualize(self):
        hand_strs = [[suit_str[suit] + ' ' + ''.join(rank_str[rank] for rank in ranks)
            for suit, ranks in enumerate(hand)] for hand in self.hands]
        for suit in range(4): print('        '  + hand_strs[1][suit])
        for suit in range(4): print('%-16s' % hand_strs[0][suit] + hand_strs[2][suit])
        for suit in range(4): print('        '  + hand_strs[3][suit])

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
