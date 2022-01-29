import sys,tty,termios,time
class _Getch:
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

def get():
    inkey = _Getch()
    while(1):
        k=inkey()
        if k!='':break
    if k=='\x1b[A':
        print("up")
    elif k=='\x1b[B':
        print("down")
    elif k=='\x1b[C':
        print("right")
    elif k=='\x1b[D':
        print("left")
    elif k=='\x1bw':
        print("up")
    elif k=='\x1ba':
        print("left")
    elif k=='\x1bs':
        print("down")
    elif k=='\x1bd':
        print("right")
    else:
        print("not an arrow key!")

def main():
    for i in range(0,20):
        get()

if __name__=='__main__':
    main()
