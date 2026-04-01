from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple, Type


Position = Tuple[int, int]


class ChessError(Exception):
    """Базовое исключение шахматного симулятора."""


class InvalidMoveError(ChessError):
    """Ошибка некорректного хода."""


@dataclass
class Move:
    """Объект хода с полной информацией для истории и отката."""

    start: Position
    end: Position
    piece_name: str
    piece_color: str
    captured_name: Optional[str] = None
    captured_color: Optional[str] = None
    promotion_from: Optional[str] = None
    promotion_to: Optional[str] = None
    is_castling: bool = False
    rook_start: Optional[Position] = None
    rook_end: Optional[Position] = None
    is_en_passant: bool = False
    en_passant_captured_pos: Optional[Position] = None
    note: str = ""

    board_before: List[List[Optional[Tuple[str, str, bool]]]] = field(default_factory=list)
    current_player_before: str = "white"
    en_passant_target_before: Optional[Position] = None
    halfmove_clock_before: int = 0
    fullmove_number_before: int = 1


class Piece:
    name = "Piece"
    symbol = "?"
    value = 0

    def __init__(self, color: str, position: Position):
        self.color = color
        self.position = position
        self.has_moved = False

    @property
    def icon(self) -> str:
        return self.symbol.upper() if self.color == "white" else self.symbol.lower()

    def enemy_color(self) -> str:
        return "black" if self.color == "white" else "white"

    def copy(self) -> "Piece":
        new_piece = type(self)(self.color, self.position)
        new_piece.has_moved = self.has_moved
        return new_piece

    def attacks(self, board: "Board") -> Set[Position]:
        return set(self.pseudo_legal_moves(board))

    def pseudo_legal_moves(self, board: "Board") -> Set[Position]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.name}({self.color}, {self.position})"


class SlidingPiece(Piece):
    directions: Tuple[Tuple[int, int], ...] = ()

    def pseudo_legal_moves(self, board: "Board") -> Set[Position]:
        moves: Set[Position] = set()
        for dr, dc in self.directions:
            r, c = self.position
            while True:
                r += dr
                c += dc
                pos = (r, c)
                if not board.in_bounds(pos):
                    break
                target = board.get_piece(pos)
                if target is None:
                    moves.add(pos)
                    continue
                if target.color != self.color:
                    moves.add(pos)
                break
        return moves


class King(Piece):
    name = "King"
    symbol = "k"
    value = 1000

    def pseudo_legal_moves(self, board: "Board") -> Set[Position]:
        moves: Set[Position] = set()
        r, c = self.position
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                pos = (r + dr, c + dc)
                if board.in_bounds(pos):
                    target = board.get_piece(pos)
                    if target is None or target.color != self.color:
                        moves.add(pos)

        # Рокировка добавляется как псевдолегальная, финальная проверка — в Board
        if not self.has_moved and not board.is_in_check(self.color):
            row = 7 if self.color == "white" else 0
            # короткая
            rook = board.get_piece((row, 7))
            if (
                isinstance(rook, Rook)
                and rook.color == self.color
                and not rook.has_moved
                and board.get_piece((row, 5)) is None
                and board.get_piece((row, 6)) is None
                and not board.is_square_attacked((row, 5), self.enemy_color())
                and not board.is_square_attacked((row, 6), self.enemy_color())
            ):
                moves.add((row, 6))
            # длинная
            rook = board.get_piece((row, 0))
            if (
                isinstance(rook, Rook)
                and rook.color == self.color
                and not rook.has_moved
                and board.get_piece((row, 1)) is None
                and board.get_piece((row, 2)) is None
                and board.get_piece((row, 3)) is None
                and not board.is_square_attacked((row, 2), self.enemy_color())
                and not board.is_square_attacked((row, 3), self.enemy_color())
            ):
                moves.add((row, 2))
        return moves


class Queen(SlidingPiece):
    name = "Queen"
    symbol = "q"
    value = 9
    directions = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )


class Rook(SlidingPiece):
    name = "Rook"
    symbol = "r"
    value = 5
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))


class Bishop(SlidingPiece):
    name = "Bishop"
    symbol = "b"
    value = 3
    directions = ((-1, -1), (-1, 1), (1, -1), (1, 1))


class Knight(Piece):
    name = "Knight"
    symbol = "n"
    value = 3

    def pseudo_legal_moves(self, board: "Board") -> Set[Position]:
        moves: Set[Position] = set()
        r, c = self.position
        for dr, dc in [
            (-2, -1), (-2, 1), (2, -1), (2, 1),
            (-1, -2), (-1, 2), (1, -2), (1, 2),
        ]:
            pos = (r + dr, c + dc)
            if board.in_bounds(pos):
                target = board.get_piece(pos)
                if target is None or target.color != self.color:
                    moves.add(pos)
        return moves


class Pawn(Piece):
    name = "Pawn"
    symbol = "p"
    value = 1

    def pseudo_legal_moves(self, board: "Board") -> Set[Position]:
        moves: Set[Position] = set()
        r, c = self.position
        direction = -1 if self.color == "white" else 1
        start_row = 6 if self.color == "white" else 1

        one = (r + direction, c)
        if board.in_bounds(one) and board.get_piece(one) is None:
            moves.add(one)
            two = (r + 2 * direction, c)
            if r == start_row and board.in_bounds(two) and board.get_piece(two) is None:
                moves.add(two)

        for dc in (-1, 1):
            attack = (r + direction, c + dc)
            if not board.in_bounds(attack):
                continue
            target = board.get_piece(attack)
            if target is not None and target.color != self.color:
                moves.add(attack)
            elif board.en_passant_target == attack:
                moves.add(attack)
        return moves

    def attacks(self, board: "Board") -> Set[Position]:
        r, c = self.position
        direction = -1 if self.color == "white" else 1
        result: Set[Position] = set()
        for dc in (-1, 1):
            pos = (r + direction, c + dc)
            if board.in_bounds(pos):
                result.add(pos)
        return result


# --- Дополнительные оригинальные фигуры ---
class Chancellor(Piece):
    """Ладья + конь."""

    name = "Chancellor"
    symbol = "c"
    value = 8

    def pseudo_legal_moves(self, board: "Board") -> Set[Position]:
        rook_like = Rook(self.color, self.position)
        rook_like.has_moved = self.has_moved
        knight_like = Knight(self.color, self.position)
        knight_like.has_moved = self.has_moved
        return rook_like.pseudo_legal_moves(board) | knight_like.pseudo_legal_moves(board)


class Archbishop(Piece):
    """Слон + конь."""

    name = "Archbishop"
    symbol = "a"
    value = 7

    def pseudo_legal_moves(self, board: "Board") -> Set[Position]:
        bishop_like = Bishop(self.color, self.position)
        bishop_like.has_moved = self.has_moved
        knight_like = Knight(self.color, self.position)
        knight_like.has_moved = self.has_moved
        return bishop_like.pseudo_legal_moves(board) | knight_like.pseudo_legal_moves(board)


class Camel(Piece):
    """Прыгающая фигура (3,1), проходит через фигуры."""

    name = "Camel"
    symbol = "m"
    value = 4

    def pseudo_legal_moves(self, board: "Board") -> Set[Position]:
        moves: Set[Position] = set()
        r, c = self.position
        for dr, dc in [
            (-3, -1), (-3, 1), (3, -1), (3, 1),
            (-1, -3), (-1, 3), (1, -3), (1, 3),
        ]:
            pos = (r + dr, c + dc)
            if board.in_bounds(pos):
                target = board.get_piece(pos)
                if target is None or target.color != self.color:
                    moves.add(pos)
        return moves


PIECE_REGISTRY: Dict[str, Type[Piece]] = {
    "K": King,
    "Q": Queen,
    "R": Rook,
    "B": Bishop,
    "N": Knight,
    "P": Pawn,
    "C": Chancellor,
    "A": Archbishop,
    "M": Camel,
}


class Board:
    def __init__(self, variant: str = "classic"):
        self.grid: List[List[Optional[Piece]]] = [[None for _ in range(8)] for _ in range(8)]
        self.variant = variant
        self.current_player = "white"
        self.move_history: List[Move] = []
        self.en_passant_target: Optional[Position] = None
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self.setup(variant)

    def setup(self, variant: str = "classic") -> None:
        self.grid = [[None for _ in range(8)] for _ in range(8)]
        self.current_player = "white"
        self.move_history.clear()
        self.en_passant_target = None
        self.halfmove_clock = 0
        self.fullmove_number = 1

        for col in range(8):
            self.place_piece(Pawn("white", (6, col)))
            self.place_piece(Pawn("black", (1, col)))

        if variant == "classic":
            back_rank = [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook]
        elif variant == "fantasy":
            back_rank = [Rook, Chancellor, Bishop, Queen, King, Archbishop, Camel, Rook]
        else:
            raise ValueError("Неизвестный вариант доски")

        for col, piece_cls in enumerate(back_rank):
            self.place_piece(piece_cls("white", (7, col)))
            self.place_piece(piece_cls("black", (0, col)))

    def snapshot(self) -> List[List[Optional[Tuple[str, str, bool]]]]:
        snap: List[List[Optional[Tuple[str, str, bool]]]] = []
        for row in self.grid:
            snap_row: List[Optional[Tuple[str, str, bool]]] = []
            for piece in row:
                if piece is None:
                    snap_row.append(None)
                else:
                    snap_row.append((piece.__class__.__name__, piece.color, piece.has_moved))
            snap.append(snap_row)
        return snap

    def restore_snapshot(self, snapshot: List[List[Optional[Tuple[str, str, bool]]]]) -> None:
        self.grid = [[None for _ in range(8)] for _ in range(8)]
        name_to_cls = {cls.__name__: cls for cls in PIECE_REGISTRY.values()}
        for r in range(8):
            for c in range(8):
                item = snapshot[r][c]
                if item is None:
                    continue
                class_name, color, has_moved = item
                piece_cls = name_to_cls[class_name]
                piece = piece_cls(color, (r, c))
                piece.has_moved = has_moved
                self.grid[r][c] = piece

    def copy(self) -> "Board":
        new_board = Board(self.variant)
        new_board.restore_snapshot(self.snapshot())
        new_board.current_player = self.current_player
        new_board.en_passant_target = self.en_passant_target
        new_board.halfmove_clock = self.halfmove_clock
        new_board.fullmove_number = self.fullmove_number
        new_board.move_history = list(self.move_history)
        return new_board

    def in_bounds(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < 8 and 0 <= c < 8

    def get_piece(self, pos: Position) -> Optional[Piece]:
        r, c = pos
        return self.grid[r][c]

    def place_piece(self, piece: Piece) -> None:
        r, c = piece.position
        self.grid[r][c] = piece

    def remove_piece(self, pos: Position) -> Optional[Piece]:
        piece = self.get_piece(pos)
        r, c = pos
        self.grid[r][c] = None
        return piece

    def move_piece_raw(self, start: Position, end: Position) -> Optional[Piece]:
        piece = self.get_piece(start)
        if piece is None:
            return None
        captured = self.remove_piece(end)
        self.remove_piece(start)
        piece.position = end
        piece.has_moved = True
        self.place_piece(piece)
        return captured

    def algebraic_to_pos(self, cell: str) -> Position:
        cell = cell.strip().lower()
        if len(cell) != 2 or cell[0] not in "abcdefgh" or cell[1] not in "12345678":
            raise ValueError(f"Некорректная клетка: {cell}")
        col = ord(cell[0]) - ord("a")
        row = 8 - int(cell[1])
        return row, col

    def pos_to_algebraic(self, pos: Position) -> str:
        r, c = pos
        return f"{chr(ord('a') + c)}{8 - r}"

    def parse_move(self, text: str) -> Tuple[Position, Position, Optional[str]]:
        parts = text.strip().lower().replace("-", " ").split()
        if len(parts) < 2:
            raise ValueError("Введите ход в формате: e2 e4 или e7 e8 q")
        start = self.algebraic_to_pos(parts[0])
        end = self.algebraic_to_pos(parts[1])
        promotion = parts[2].upper() if len(parts) >= 3 else None
        return start, end, promotion

    def locate_king(self, color: str) -> Position:
        for r in range(8):
            for c in range(8):
                piece = self.grid[r][c]
                if isinstance(piece, King) and piece.color == color:
                    return (r, c)
        raise ChessError(f"Король цвета {color} не найден")

    def pieces(self, color: Optional[str] = None) -> Iterable[Piece]:
        for row in self.grid:
            for piece in row:
                if piece is None:
                    continue
                if color is None or piece.color == color:
                    yield piece

    def is_square_attacked(self, pos: Position, by_color: str) -> bool:
        for piece in self.pieces(by_color):
            # Для короля используем только соседние клетки атаки,
            # чтобы избежать рекурсии king -> is_in_check -> is_square_attacked -> king
            if isinstance(piece, King):
                r, c = piece.position
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        if (r + dr, c + dc) == pos:
                            return True
                continue

            if pos in piece.attacks(self):
                return True
        return False

    def threatened_pieces(self, color: str) -> List[Piece]:
        result: List[Piece] = []
        enemy = "black" if color == "white" else "white"
        for piece in self.pieces(color):
            if self.is_square_attacked(piece.position, enemy):
                result.append(piece)
        return result

    def is_in_check(self, color: str) -> bool:
        king_pos = self.locate_king(color)
        enemy = "black" if color == "white" else "white"
        return self.is_square_attacked(king_pos, enemy)

    def legal_moves_for_piece(self, pos: Position) -> Set[Position]:
        piece = self.get_piece(pos)
        if piece is None:
            return set()
        moves: Set[Position] = set()
        for target in piece.pseudo_legal_moves(self):
            clone = self.copy()
            clone._apply_move_no_validation(pos, target, promotion_choice="Q")
            if not clone.is_in_check(piece.color):
                moves.add(target)
        return moves

    def all_legal_moves(self, color: str) -> List[Tuple[Position, Position]]:
        result: List[Tuple[Position, Position]] = []
        for piece in self.pieces(color):
            for target in self.legal_moves_for_piece(piece.position):
                result.append((piece.position, target))
        return result

    def _apply_move_no_validation(self, start: Position, end: Position, promotion_choice: Optional[str]) -> Move:
        piece = self.get_piece(start)
        if piece is None:
            raise InvalidMoveError("На стартовой клетке нет фигуры")

        move = Move(
            start=start,
            end=end,
            piece_name=piece.name,
            piece_color=piece.color,
            board_before=self.snapshot(),
            current_player_before=self.current_player,
            en_passant_target_before=self.en_passant_target,
            halfmove_clock_before=self.halfmove_clock,
            fullmove_number_before=self.fullmove_number,
        )

        target_before = self.get_piece(end)
        if target_before is not None:
            move.captured_name = target_before.name
            move.captured_color = target_before.color

        # Взятие на проходе
        if isinstance(piece, Pawn) and self.en_passant_target == end and target_before is None and start[1] != end[1]:
            captured_pos = (start[0], end[1])
            captured_piece = self.remove_piece(captured_pos)
            if captured_piece is not None:
                move.captured_name = captured_piece.name
                move.captured_color = captured_piece.color
                move.is_en_passant = True
                move.en_passant_captured_pos = captured_pos
                move.note = "en passant"

        # Рокировка
        if isinstance(piece, King) and abs(start[1] - end[1]) == 2:
            row = start[0]
            move.is_castling = True
            if end[1] == 6:
                rook_start, rook_end = (row, 7), (row, 5)
                move.note = "short castling"
            else:
                rook_start, rook_end = (row, 0), (row, 3)
                move.note = "long castling"
            move.rook_start = rook_start
            move.rook_end = rook_end
            self.move_piece_raw(rook_start, rook_end)

        captured_after = self.move_piece_raw(start, end)
        if captured_after is not None and move.captured_name is None:
            move.captured_name = captured_after.name
            move.captured_color = captured_after.color

        moved_piece = self.get_piece(end)
        assert moved_piece is not None

        # Пешка: en passant target
        self.en_passant_target = None
        if isinstance(moved_piece, Pawn) and abs(start[0] - end[0]) == 2:
            self.en_passant_target = ((start[0] + end[0]) // 2, start[1])

        # Превращение пешки
        if isinstance(moved_piece, Pawn) and end[0] in (0, 7):
            promotion_letter = (promotion_choice or "Q").upper()
            promotion_cls = PIECE_REGISTRY.get(promotion_letter, Queen)
            if promotion_cls is Pawn or promotion_cls is King:
                promotion_cls = Queen
            move.promotion_from = "Pawn"
            move.promotion_to = promotion_cls.__name__
            promoted_piece = promotion_cls(moved_piece.color, end)
            promoted_piece.has_moved = True
            self.grid[end[0]][end[1]] = promoted_piece
            move.note = (move.note + "; " if move.note else "") + f"promotion to {promotion_cls.__name__}"

        if move.captured_name or isinstance(piece, Pawn):
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        self.current_player = "black" if self.current_player == "white" else "white"
        if self.current_player == "white":
            self.fullmove_number += 1

        self.move_history.append(move)
        return move

    def make_move(self, start: Position, end: Position, promotion_choice: Optional[str] = None) -> Move:
        piece = self.get_piece(start)
        if piece is None:
            raise InvalidMoveError("На указанной клетке нет фигуры")
        if piece.color != self.current_player:
            raise InvalidMoveError("Сейчас ход другого игрока")
        legal = self.legal_moves_for_piece(start)
        if end not in legal:
            raise InvalidMoveError(
                f"Недопустимый ход для {piece.name}: {self.pos_to_algebraic(start)} -> {self.pos_to_algebraic(end)}"
            )
        return self._apply_move_no_validation(start, end, promotion_choice)

    def undo(self, steps: int = 1) -> None:
        if steps < 1:
            raise ValueError("Количество откатов должно быть >= 1")
        if steps > len(self.move_history):
            raise InvalidMoveError("Недостаточно ходов в истории")

        for _ in range(steps):
            move = self.move_history.pop()
            self.restore_snapshot(move.board_before)
            self.current_player = move.current_player_before
            self.en_passant_target = move.en_passant_target_before
            self.halfmove_clock = move.halfmove_clock_before
            self.fullmove_number = move.fullmove_number_before

    def game_state(self) -> str:
        enemy = self.current_player
        moves = self.all_legal_moves(enemy)
        in_check = self.is_in_check(enemy)
        if not moves and in_check:
            return f"checkmate_{enemy}"
        if not moves and not in_check:
            return "stalemate"
        return "normal"

    def render(self, highlight_moves: Optional[Set[Position]] = None) -> str:
        highlight_moves = highlight_moves or set()
        threatened_white = {p.position for p in self.threatened_pieces("white")}
        threatened_black = {p.position for p in self.threatened_pieces("black")}
        check_white = self.locate_king("white") if self.is_in_check("white") else None
        check_black = self.locate_king("black") if self.is_in_check("black") else None

        lines: List[str] = []
        lines.append("    a  b  c  d  e  f  g  h")
        lines.append("  +------------------------+")
        for r in range(8):
            row_cells: List[str] = []
            for c in range(8):
                pos = (r, c)
                piece = self.get_piece(pos)
                char = piece.icon if piece else "."

                if pos == check_white or pos == check_black:
                    cell = f"!{char}"
                elif pos in highlight_moves:
                    cell = f"*{char}"
                elif pos in threatened_white or pos in threatened_black:
                    cell = f"?{char}"
                else:
                    cell = f" {char}"
                row_cells.append(cell)
            lines.append(f"{8 - r} |" + "".join(row_cells) + f" | {8 - r}")
        lines.append("  +------------------------+")
        lines.append("    a  b  c  d  e  f  g  h")
        lines.append(
            f"Ход: {'белые' if self.current_player == 'white' else 'чёрные'} | "
            f"Вариант: {self.variant} | Полный ход: {self.fullmove_number}"
        )
        lines.append("Легенда: * допустимый ход, ? фигура под боем, ! король под шахом")
        return "\n".join(lines)


class Game:
    def __init__(self, variant: str = "classic"):
        self.board = Board(variant=variant)

    def print_help(self) -> None:
        help_text = """
Команды:
  move e2 e4           - сделать ход
  move e7 e8 q         - ход с превращением пешки
  hint e2              - показать допустимые ходы фигуры
  undo                 - откатить 1 ход
  undo 3               - откатить 3 хода
  board                - вывести доску
  restart              - начать заново текущий вариант
  variant classic      - классические шахматы
  variant fantasy      - шахматы с новыми фигурами C/A/M
  pieces               - список фигур варианта fantasy
  help                 - справка
  exit                 - выход
        """.strip()
        print(help_text)

    def print_fantasy_info(self) -> None:
        print(
            "Новые фигуры:\n"
            "  C / c = Chancellor (ладья + конь)\n"
            "  A / a = Archbishop (слон + конь)\n"
            "  M / m = Camel (прыжок на 3x1, перепрыгивает фигуры)"
        )

    def status_report(self) -> None:
        if self.board.is_in_check(self.board.current_player):
            print("Шах!")
        threatened = self.board.threatened_pieces(self.board.current_player)
        if threatened:
            names = ", ".join(
                f"{p.name}({self.board.pos_to_algebraic(p.position)})" for p in threatened
            )
            print(f"Под боем: {names}")

        state = self.board.game_state()
        if state.startswith("checkmate"):
            loser = "белые" if state.endswith("white") else "чёрные"
            winner = "чёрные" if loser == "белые" else "белые"
            print(f"Мат! Проиграли {loser}. Победили {winner}.")
        elif state == "stalemate":
            print("Пат! Ничья.")

    def run(self) -> None:
        print("=== Шахматный симулятор (ООП-версия) ===")
        self.print_help()
        print(self.board.render())
        self.status_report()

        while True:
            try:
                command = input("\n> ").strip()
                if not command:
                    continue

                parts = command.split()
                action = parts[0].lower()

                if action == "exit":
                    print("Выход из игры.")
                    break

                if action == "help":
                    self.print_help()
                    continue

                if action == "board":
                    print(self.board.render())
                    self.status_report()
                    continue

                if action == "pieces":
                    self.print_fantasy_info()
                    continue

                if action == "restart":
                    variant = self.board.variant
                    self.board = Board(variant=variant)
                    print(self.board.render())
                    self.status_report()
                    continue

                if action == "variant":
                    if len(parts) != 2:
                        print("Использование: variant classic|fantasy")
                        continue
                    self.board = Board(parts[1].lower())
                    print(self.board.render())
                    self.status_report()
                    continue

                if action == "hint":
                    if len(parts) != 2:
                        print("Использование: hint e2")
                        continue
                    pos = self.board.algebraic_to_pos(parts[1])
                    piece = self.board.get_piece(pos)
                    if piece is None:
                        print("На этой клетке нет фигуры.")
                        continue
                    legal_moves = self.board.legal_moves_for_piece(pos)
                    moves_text = ", ".join(sorted(self.board.pos_to_algebraic(m) for m in legal_moves)) or "нет"
                    print(self.board.render(highlight_moves=legal_moves))
                    print(f"Допустимые ходы для {piece.name} {parts[1]}: {moves_text}")
                    continue

                if action == "undo":
                    steps = int(parts[1]) if len(parts) > 1 else 1
                    self.board.undo(steps)
                    print(self.board.render())
                    self.status_report()
                    continue

                if action == "move":
                    if len(parts) < 3:
                        print("Использование: move e2 e4 [q]")
                        continue
                    start, end = self.board.algebraic_to_pos(parts[1]), self.board.algebraic_to_pos(parts[2])
                    promotion = parts[3].upper() if len(parts) > 3 else None
                    move = self.board.make_move(start, end, promotion)
                    print(self.board.render())
                    print(
                        f"Ход выполнен: {self.board.pos_to_algebraic(move.start)} -> "
                        f"{self.board.pos_to_algebraic(move.end)}"
                        + (f" ({move.note})" if move.note else "")
                    )
                    self.status_report()
                    continue

                # Позволяем писать ход сразу: e2 e4
                if len(parts) in (2, 3):
                    start, end, promotion = self.board.parse_move(command)
                    move = self.board.make_move(start, end, promotion)
                    print(self.board.render())
                    print(
                        f"Ход выполнен: {self.board.pos_to_algebraic(move.start)} -> "
                        f"{self.board.pos_to_algebraic(move.end)}"
                        + (f" ({move.note})" if move.note else "")
                    )
                    self.status_report()
                    continue

                print("Неизвестная команда. Введите help.")

            except (ValueError, InvalidMoveError, ChessError) as exc:
                print(f"Ошибка: {exc}")
            except KeyboardInterrupt:
                print("\nВыход из игры.")
                break


def implementation_plan() -> str:
    return (
        "План реализации:\n"
        "1. Создать базовый класс Piece и наследников для всех фигур.\n"
        "2. Вынести состояние доски в класс Board: клетки, текущий игрок, история ходов.\n"
        "3. Представить каждый ход объектом Move с данными для отката.\n"
        "4. Делегировать вычисление возможных ходов самим фигурам.\n"
        "5. Реализовать фильтрацию псевдолегальных ходов через проверку шаха.\n"
        "6. Добавить специальные правила: рокировка, взятие на проходе, превращение пешки.\n"
        "7. Реализовать подсказки ходов и подсветку фигур под боем.\n"
        "8. Добавить новые фигуры через расширение иерархии без переписывания Board.\n"
        "9. Хранить историю ходов и реализовать откат на любое число полуходов.\n"
        "10. Сделать консольный интерфейс команд для демонстрации работы."
    )


if __name__ == "__main__":
    print(implementation_plan())
    game = Game(variant="classic")
    game.run()
