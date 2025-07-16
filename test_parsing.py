import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='Your name')
    parser.add_argument('--age', type=str, help='Your age')
    args = parser.parse_args()

    print(f'Hello {args.name}, you are {args.age} years old !')