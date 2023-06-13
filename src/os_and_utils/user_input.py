import sys,tty,termios,time
class _Getch:
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

def getch():
    inkey = _Getch()
    while(1):
        k=inkey()
        if k!='':break
    print("key", k)

def main():
    while True:
        getch()

if __name__=='__main__':
    main()
