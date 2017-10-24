from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()
def main():
    print similar("A finite-state morphological grammar of Hebrew", "A Finite-State Morphological Grammar Of Hebrew")
if __name__ == '__main__':
    main()