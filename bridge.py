import random

suit_str = 'SHDC'
rank_str = '  23456789TJQKA'
player_str = ['West', 'North', 'East', 'South']

def move_to_str(move):
    return '%s%s' % (suit_str[move[0]], rank_str[move[1]])

def move_to_int(move):
    return move[0] * 13 + move[1] - 2

def int_to_move(i):
    return i // 13, i % 13 + 2

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
        self.current_player = 0
        self.cards_in_trick = 0
        self.current_suit = None
        self.current_high = None
        self.current_high_player = None
        self.score = 0 # Number of tricks won by players 0 and 2
        self.move_to_str = move_to_str
        self.move_to_int = move_to_int
        self.int_to_move = int_to_move

    def get_moves(self):
        if self.cards_in_trick == 0 or self.hands[self.current_player][self.current_suit] == []: suits = range(4)
        else: suits = [self.current_suit]
        res = []
        for suit in suits:
            for rank in self.hands[self.current_player][suit]:
                res.append((suit, rank))
        return res

    def make_move(self, move):
        suit, rank = move
        self.hands[self.current_player][suit].remove(rank)
        if self.cards_in_trick == 0: self.current_suit = suit
        if self.cards_in_trick == 0 or suit == self.current_suit and rank > self.current_high:
            self.current_high = rank
            self.current_high_player = self.current_player
        if self.cards_in_trick == 3:
            if self.current_high_player % 2 == 0: self.score += 1
            trick_won = (self.current_player - self.current_high_player) % 2 == 0
            self.current_player = self.current_high_player
            self.cards_in_trick = 0
            self.current_suit = None
            self.current_high = None
            self.current_high_player = None
            return trick_won
        self.current_player = (self.current_player + 1) % 4
        self.cards_in_trick += 1
        return False

    def visualize(self, verbose = False):
        hand_strs = [[suit_str[suit] + ' ' + ''.join(rank_str[rank] for rank in ranks)
            for suit, ranks in enumerate(hand)] for hand in self.hands]
        for suit in range(4): print('        '  + hand_strs[1][suit])
        for suit in range(4): print('%-16s' % hand_strs[0][suit] + hand_strs[2][suit])
        for suit in range(4): print('        '  + hand_strs[3][suit])
        if verbose:
            if self.cards_in_trick == 0: print('%s to lead' % player_str[self.current_player])
            else: print('%s to play, %s in hand, %s is led, %s from %s currently high' %
                       (player_str[self.current_player],
                        [None, '2nd', '3rd', '4th'][self.cards_in_trick],
                        suit_str[self.current_suit],
                        rank_str[self.current_high],
                        player_str[self.current_high_player]))

    # Converts the position to a tensor for use with TensorFlow (currently a vector)
    def to_tensor(self):
        res = []
        for i in range(4):
            hand = self.hands[(self.current_player + i) % 4]
            for suit in range(4):
                for rank in range(2, 15):
                    res.append(1 if rank in hand[suit] else 0)
        for i in range(4): res.append(1 if self.cards_in_trick == i else 0)
        for suit in range(4): res.append(1 if self.current_suit == suit else 0)
        for rank in range(2, 15): res.append(1 if self.current_high == rank else 0)
        for i in range(1, 4): res.append(1 if self.current_high_player == (self.current_player + i) % 4 else 0)
        return res

    @classmethod
    def from_tensor(cls, tensor):
        hands = []
        it = iter(tensor)
        for i in range(4):
            hand = [[] for suit in range(4)]
            for suit in range(4):
                for rank in range(2, 15):
                    if next(it) == 1: hand[suit].append(rank)
                hand[suit] = hand[suit][::-1]
            hands.append(hand)
        pos = cls(hands = hands)
        for i in range(4):
            if next(it) == 1: pos.cards_in_trick = i
        for suit in range(4):
            if next(it) == 1: pos.current_suit = suit
        for rank in range(2, 15):
            if next(it) == 1: pos.current_high = rank
        for i in range(1, 4):
            if next(it) == 1: pos.current_high_player = i
        return pos
