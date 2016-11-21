def canIWin(maxChoosableInteger, desiredTotal):
    """
    :type maxChoosableInteger: int
    :type desiredTotal: int
    :rtype: bool
    """
    win_map = [[-1 for j in range(desiredTotal+maxChoosableInteger+1)] for i in range(desiredTotal+maxChoosableInteger+1)]
    for i in range(desiredTotal - maxChoosableInteger, desiredTotal+maxChoosableInteger+1):
        for j in range(desiredTotal+maxChoosableInteger+1):
            win_map[i][j] = 1

    def recur_win(x, y, options):

        if win_map[x][y] != -1:
            if win_map[x][y] == 0:
                return False
            if win_map[x][y] == 1:
                return True

        else:
            length = len(options)
            res = False
            for i in range(length):
                choice = options.pop(i)
                if recur_win(y, x+choice, options) == False:
                    win_map[x][y] = 1
                    return True
                options.insert(i, choice)

            win_map[x][y] = 0
            return False

    lst = list(range(maxChoosableInteger + 1))
    print(win_map)
    return recur_win(0, 0, lst)

print(canIWin(10, 11))
