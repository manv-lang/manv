from __future__ import annotations

from . import ast
from .diagnostics import ManvError, Span, diag
from .tokens import Token

C_DECL_TYPES = {"int", "str", "array", "map", "u8", "usize", "float", "bool"}


class Parser:
    def __init__(self, tokens: list[Token], file: str, source_lines: list[str]):
        self.tokens = tokens
        self.file = file
        self.source_lines = source_lines
        self.pos = 0

    def parse(self) -> ast.Program:
        declarations: list[object] = []
        statements: list[object] = []
        while not self._is("EOF"):
            self._consume_newlines()
            if self._is("EOF"):
                break
            if self._match_keyword("fn"):
                declarations.append(self._parse_fn_decl())
            elif self._match_keyword("type"):
                declarations.append(self._parse_type_decl())
            elif self._match_keyword("impl"):
                declarations.append(self._parse_impl_decl())
            elif self._match_keyword("macro"):
                declarations.append(self._parse_macro_decl_stub())
            else:
                statements.append(self._parse_statement())
        span = Span(self.file, 1, 1)
        return ast.Program(declarations=declarations, statements=statements, span=span)

    def _parse_fn_decl(self) -> ast.FnDecl:
        fn_tok = self._prev()
        name_tok = self._expect("IDENT", message="expected function name")
        self._expect_op("(")
        params: list[ast.Param] = []
        if not self._is_op(")"):
            while True:
                p_tok = self._expect("IDENT", message="expected parameter name")
                p_type = None
                if self._match_op(":"):
                    p_type = self._parse_type(stop_ops={",", ")"})
                params.append(ast.Param(name=p_tok.lexeme, type_name=p_type, span=self._span(p_tok)))
                if self._match_op(","):
                    continue
                break
        self._expect_op(")")
        return_type = None
        if self._match_op("->"):
            return_type = self._parse_type(stop_ops={":"})
        self._expect_op(":")
        body = self._parse_block()
        return ast.FnDecl(
            name=name_tok.lexeme,
            params=params,
            return_type=return_type,
            body=body,
            span=self._span(fn_tok),
        )

    def _parse_type_decl(self) -> ast.TypeDecl:
        type_tok = self._prev()
        name_tok = self._expect("IDENT", message="expected type name")
        base_name: str | None = None
        if self._match_op("("):
            base_tok = self._expect("IDENT", message="expected base type name")
            base_name = base_tok.lexeme
            self._expect_op(")")
        self._expect_op(":")
        methods = self._parse_method_block("type")
        return ast.TypeDecl(name=name_tok.lexeme, base_name=base_name, methods=methods, span=self._span(type_tok))

    def _parse_impl_decl(self) -> ast.ImplDecl:
        impl_tok = self._prev()
        target_tok = self._expect("IDENT", message="expected impl target type")
        self._expect_op(":")
        methods = self._parse_method_block("impl")
        return ast.ImplDecl(target=target_tok.lexeme, methods=methods, span=self._span(impl_tok))

    def _parse_method_block(self, owner: str) -> list[ast.FnDecl]:
        self._consume_required_newline("expected newline after ':'")
        self._expect("INDENT", message="expected indented block")
        methods: list[ast.FnDecl] = []
        while not self._is("DEDENT") and not self._is("EOF"):
            self._consume_newlines()
            if self._is("DEDENT") or self._is("EOF"):
                break
            if not self._match_keyword("fn"):
                self._error("E1006", f"expected 'fn' inside {owner} block", self._current())
            methods.append(self._parse_fn_decl())
        self._expect("DEDENT", message="expected dedent")
        return methods

    def _parse_macro_decl_stub(self) -> ast.MacroDeclStub:
        macro_tok = self._prev()
        name_tok = self._expect("IDENT", message="expected macro name")
        params: list[str] = []
        if self._match_op("("):
            if not self._is_op(")"):
                while True:
                    p = self._expect("IDENT", message="expected macro parameter")
                    params.append(p.lexeme)
                    if self._match_op(","):
                        continue
                    break
            self._expect_op(")")
        self._expect_op(":")
        body = self._parse_stub_block()
        return ast.MacroDeclStub(name=name_tok.lexeme, params=params, body=body, span=self._span(macro_tok))

    def _parse_statement(self) -> object:
        if self._is_c_declaration_start():
            stmt = self._parse_c_decl_stmt()
            self._consume_required_newline("expected newline after declaration")
            return stmt
        if self._match_keyword("let"):
            stmt = self._parse_let_stmt()
            self._consume_required_newline("expected newline after let statement")
            return stmt
        if self._match_keyword("return"):
            stmt = self._parse_return_stmt()
            self._consume_required_newline("expected newline after return")
            return stmt
        if self._match_keyword("raise"):
            stmt = self._parse_raise_stmt()
            self._consume_required_newline("expected newline after raise")
            return stmt
        if self._match_keyword("break"):
            tok = self._prev()
            self._consume_required_newline("expected newline after break")
            return ast.BreakStmt(span=self._span(tok))
        if self._match_keyword("continue"):
            tok = self._prev()
            self._consume_required_newline("expected newline after continue")
            return ast.ContinueStmt(span=self._span(tok))
        if self._match_keyword("if"):
            return self._parse_if_stmt()
        if self._match_keyword("while"):
            return self._parse_while_stmt()
        if self._match_keyword("try"):
            return self._parse_try_stmt()
        if self._match_keyword("gpu"):
            token = self._prev()
            detail = self._collect_until_newline()
            self._consume_required_newline("expected newline")
            return ast.UnsupportedStmt(feature="gpu", detail=detail, span=self._span(token))
        if self._match_keyword("memory"):
            token = self._prev()
            detail = self._collect_until_newline()
            self._consume_required_newline("expected newline")
            return ast.UnsupportedStmt(feature="memory", detail=detail, span=self._span(token))

        if self._is("IDENT") and self._peek(1).kind == "OP" and self._peek(1).lexeme == "=":
            name_tok = self._advance()
            assign_tok = self._advance()
            value = self._parse_expression()
            self._consume_required_newline("expected newline after assignment")
            return ast.AssignStmt(name=name_tok.lexeme, value=value, span=self._span(assign_tok))

        attr_assign = self._try_parse_attr_assignment()
        if attr_assign is not None:
            self._consume_required_newline("expected newline after attribute assignment")
            return attr_assign

        index_assign = self._try_parse_index_assignment()
        if index_assign is not None:
            self._consume_required_newline("expected newline after index assignment")
            return index_assign

        expr = self._parse_expression()
        self._consume_required_newline("expected newline after expression")
        return ast.ExprStmt(expr=expr, span=self._span_from_expr(expr))

    def _parse_let_stmt(self) -> ast.LetStmt:
        let_tok = self._prev()
        if self._is("IDENT") and self._current().lexeme == "mut" and self._peek(1).kind == "IDENT":
            self._error("E1010", "`mut` declarations are not supported; use `let <name> = ...`", self._current())
        name_tok = self._expect("IDENT", message="expected variable name")
        type_name = None
        value = None
        if self._match_op(":"):
            type_name = self._parse_type(stop_ops={"=", "\n"})
        if self._match_op("="):
            value = self._parse_expression()
        return ast.LetStmt(
            name=name_tok.lexeme,
            type_name=type_name,
            value=value,
            span=self._span(let_tok),
        )

    def _parse_c_decl_stmt(self) -> ast.LetStmt:
        type_tok = self._advance()
        type_name = type_tok.lexeme
        name_tok = self._expect("IDENT", message="expected variable name")
        array_size: object | None = None

        if self._match_op("["):
            array_size = self._parse_expression()
            self._expect_op("]")
            if type_name != "array":
                type_name = f"array[{type_name}]"

        value = None
        if self._match_op("="):
            value = self._parse_expression()

        return ast.LetStmt(
            name=name_tok.lexeme,
            type_name=type_name,
            value=value,
            span=self._span(type_tok),
            array_size=array_size,
        )

    def _parse_return_stmt(self) -> ast.ReturnStmt:
        tok = self._prev()
        if self._is("NEWLINE"):
            return ast.ReturnStmt(value=None, span=self._span(tok))
        value = self._parse_expression()
        return ast.ReturnStmt(value=value, span=self._span(tok))

    def _parse_raise_stmt(self) -> ast.RaiseStmt:
        tok = self._prev()
        if self._is("NEWLINE"):
            return ast.RaiseStmt(value=None, span=self._span(tok))
        value = self._parse_expression()
        return ast.RaiseStmt(value=value, span=self._span(tok))

    def _parse_try_stmt(self) -> ast.TryStmt:
        try_tok = self._prev()
        self._expect_op(":")
        try_body = self._parse_block()

        except_clauses: list[ast.ExceptClause] = []
        else_body: list[object] = []
        finally_body: list[object] = []

        while self._match_keyword("except"):
            ex_tok = self._prev()
            type_tok = self._expect("IDENT", message="expected exception type name")
            bind_name: str | None = None
            if self._match_keyword("as"):
                bind_tok = self._expect("IDENT", message="expected exception bind variable")
                bind_name = bind_tok.lexeme
            self._expect_op(":")
            body = self._parse_block()
            except_clauses.append(
                ast.ExceptClause(type_name=type_tok.lexeme, bind_name=bind_name, body=body, span=self._span(ex_tok))
            )

        if self._match_keyword("else"):
            self._expect_op(":")
            else_body = self._parse_block()

        if self._match_keyword("finally"):
            self._expect_op(":")
            finally_body = self._parse_block()

        if not except_clauses and not finally_body:
            self._error("E1011", "try statement requires at least one except or finally", try_tok)

        return ast.TryStmt(
            try_body=try_body,
            except_clauses=except_clauses,
            else_body=else_body,
            finally_body=finally_body,
            span=self._span(try_tok),
        )

    def _parse_if_stmt(self) -> ast.IfStmt:
        if_tok = self._prev()
        condition = self._parse_expression()
        self._expect_op(":")
        then_body = self._parse_block()
        else_body: list[object] = []
        if self._match_keyword("else"):
            self._expect_op(":")
            else_body = self._parse_block()
        return ast.IfStmt(condition=condition, then_body=then_body, else_body=else_body, span=self._span(if_tok))

    def _parse_while_stmt(self) -> ast.WhileStmt:
        while_tok = self._prev()
        condition = self._parse_expression()
        self._expect_op(":")
        body = self._parse_block()
        return ast.WhileStmt(condition=condition, body=body, span=self._span(while_tok))

    def _parse_block(self) -> list[object]:
        self._consume_required_newline("expected newline after ':'")
        self._expect("INDENT", message="expected indented block")
        body: list[object] = []
        while not self._is("DEDENT") and not self._is("EOF"):
            self._consume_newlines()
            if self._is("DEDENT") or self._is("EOF"):
                break
            body.append(self._parse_statement())
        self._expect("DEDENT", message="expected dedent")
        return body

    def _parse_stub_block(self) -> list[str]:
        self._consume_required_newline("expected newline after ':'")
        self._expect("INDENT", message="expected indented block")
        lines: list[str] = []
        current: list[str] = []
        while not self._is("DEDENT") and not self._is("EOF"):
            tok = self._advance()
            if tok.kind == "NEWLINE":
                text = " ".join(current).strip()
                if text:
                    lines.append(text)
                current = []
                continue
            current.append(tok.lexeme)
        if current:
            lines.append(" ".join(current).strip())
        self._expect("DEDENT", message="expected dedent")
        return lines

    def _parse_type(self, stop_ops: set[str]) -> str:
        parts: list[str] = []
        bracket_depth = 0
        while True:
            tok = self._current()
            if tok.kind == "NEWLINE" and bracket_depth == 0:
                break
            if tok.kind == "EOF":
                break
            if tok.kind == "OP":
                if tok.lexeme in {"[", "{", "("}:
                    bracket_depth += 1
                elif tok.lexeme in {"]", "}", ")"}:
                    bracket_depth = max(0, bracket_depth - 1)
                if bracket_depth == 0 and tok.lexeme in stop_ops:
                    break
            parts.append(self._advance().lexeme)
        result = "".join(parts).strip()
        if not result:
            tok = self._current()
            self._error("E1001", "expected type annotation", tok)
        return result

    def _parse_expression(self) -> object:
        return self._parse_or()

    def _parse_or(self) -> object:
        expr = self._parse_and()
        while self._match_logic_op("or"):
            op = self._prev().lexeme
            right = self._parse_and()
            expr = ast.BinaryExpr(left=expr, op=op, right=right, span=self._span_from_expr(expr))
        return expr

    def _parse_and(self) -> object:
        expr = self._parse_equality()
        while self._match_logic_op("and"):
            op = self._prev().lexeme
            right = self._parse_equality()
            expr = ast.BinaryExpr(left=expr, op=op, right=right, span=self._span_from_expr(expr))
        return expr

    def _parse_equality(self) -> object:
        expr = self._parse_comparison()
        while True:
            if self._match_op("=="):
                op = self._prev().lexeme
            elif self._match_op("!="):
                op = self._prev().lexeme
            else:
                break
            right = self._parse_comparison()
            expr = ast.BinaryExpr(left=expr, op=op, right=right, span=self._span_from_expr(expr))
        return expr

    def _parse_comparison(self) -> object:
        expr = self._parse_term()
        while True:
            if self._match_op("<"):
                op = self._prev().lexeme
            elif self._match_op("<="):
                op = self._prev().lexeme
            elif self._match_op(">"):
                op = self._prev().lexeme
            elif self._match_op(">="):
                op = self._prev().lexeme
            else:
                break
            right = self._parse_term()
            expr = ast.BinaryExpr(left=expr, op=op, right=right, span=self._span_from_expr(expr))
        return expr

    def _parse_term(self) -> object:
        expr = self._parse_factor()
        while True:
            if self._match_op("+"):
                op = self._prev().lexeme
            elif self._match_op("-"):
                op = self._prev().lexeme
            else:
                break
            right = self._parse_factor()
            expr = ast.BinaryExpr(left=expr, op=op, right=right, span=self._span_from_expr(expr))
        return expr

    def _parse_factor(self) -> object:
        expr = self._parse_unary()
        while True:
            if self._match_op("*"):
                op = self._prev().lexeme
            elif self._match_op("/"):
                op = self._prev().lexeme
            elif self._match_op("%"):
                op = self._prev().lexeme
            else:
                break
            right = self._parse_unary()
            expr = ast.BinaryExpr(left=expr, op=op, right=right, span=self._span_from_expr(expr))
        return expr

    def _parse_unary(self) -> object:
        if self._match_op("-"):
            op_tok = self._prev()
            right = self._parse_unary()
            return ast.UnaryExpr(op="-", expr=right, span=self._span(op_tok))
        if self._match_op("!"):
            op_tok = self._prev()
            right = self._parse_unary()
            return ast.UnaryExpr(op="!", expr=right, span=self._span(op_tok))
        if self._match_keyword("not"):
            op_tok = self._prev()
            right = self._parse_unary()
            return ast.UnaryExpr(op="not", expr=right, span=self._span(op_tok))
        return self._parse_postfix()

    def _parse_postfix(self) -> object:
        expr = self._parse_primary()
        while True:
            if self._match_op("("):
                args: list[object] = []
                if not self._is_op(")"):
                    while True:
                        args.append(self._parse_expression())
                        if self._match_op(","):
                            continue
                        break
                close = self._expect_op(")")
                expr = ast.CallExpr(callee=expr, args=args, span=self._span(close))
                continue
            if self._match_op("["):
                idx = self._parse_expression()
                close = self._expect_op("]")
                expr = ast.IndexExpr(value=expr, index=idx, span=self._span(close))
                continue
            if self._match_op("."):
                attr = self._expect("IDENT", message="expected attribute name")
                expr = ast.AttributeExpr(value=expr, attr=attr.lexeme, span=self._span(attr))
                continue
            break
        return expr

    def _parse_primary(self) -> object:
        tok = self._current()
        if self._match("NUMBER"):
            if "." in tok.lexeme:
                return ast.LiteralExpr(value=float(tok.lexeme), literal_type="float", span=self._span(tok))
            return ast.LiteralExpr(value=int(tok.lexeme), literal_type="int", span=self._span(tok))
        if self._match("STRING"):
            return ast.LiteralExpr(value=tok.lexeme, literal_type="str", span=self._span(tok))
        if self._match_keyword("true"):
            return ast.LiteralExpr(value=True, literal_type="bool", span=self._span(tok))
        if self._match_keyword("false"):
            return ast.LiteralExpr(value=False, literal_type="bool", span=self._span(tok))
        if self._match_keyword("none"):
            return ast.LiteralExpr(value=None, literal_type="none", span=self._span(tok))
        if self._match("IDENT"):
            return ast.IdentifierExpr(name=tok.lexeme, span=self._span(tok))
        if self._match_op("("):
            expr = self._parse_expression()
            self._expect_op(")")
            return expr
        if self._match_op("["):
            elements: list[object] = []
            if not self._is_op("]"):
                while True:
                    elements.append(self._parse_expression())
                    if self._match_op(","):
                        continue
                    break
            close = self._expect_op("]")
            return ast.ArrayExpr(elements=elements, span=self._span(close))
        if self._match_op("{"):
            entries: list[tuple[object, object]] = []
            if not self._is_op("}"):
                while True:
                    key = self._parse_expression()
                    self._expect_op(":")
                    value = self._parse_expression()
                    entries.append((key, value))
                    if self._match_op(","):
                        continue
                    break
            close = self._expect_op("}")
            return ast.MapExpr(entries=entries, span=self._span(close))
        self._error("E1002", "expected expression", tok)
        return ast.LiteralExpr(value=None, literal_type="none", span=self._span(tok))

    def _collect_until_newline(self) -> str:
        parts: list[str] = []
        while not self._is("NEWLINE") and not self._is("EOF"):
            parts.append(self._advance().lexeme)
        return " ".join(parts)

    def _consume_newlines(self) -> None:
        while self._match("NEWLINE"):
            pass

    def _consume_required_newline(self, message: str) -> None:
        if not self._match("NEWLINE"):
            self._error("E1003", message, self._current())
        self._consume_newlines()

    def _error(self, code: str, message: str, token: Token) -> None:
        line_text = ""
        if 1 <= token.line <= len(self.source_lines):
            line_text = self.source_lines[token.line - 1]
        raise ManvError(diag(code, message, self.file, token.line, token.column, line_text))

    def _span(self, token: Token) -> Span:
        return Span(self.file, token.line, token.column)

    def _span_from_expr(self, expr: object) -> Span:
        expr_span = getattr(expr, "span", None)
        if isinstance(expr_span, Span):
            return expr_span
        tok = self._current()
        return Span(self.file, tok.line, tok.column)

    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _peek(self, offset: int) -> Token:
        index = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[index]

    def _prev(self) -> Token:
        return self.tokens[self.pos - 1]

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _is(self, kind: str) -> bool:
        return self._current().kind == kind

    def _is_c_declaration_start(self) -> bool:
        tok = self._current()
        next_tok = self._peek(1)
        return tok.kind == "KEYWORD" and tok.lexeme in C_DECL_TYPES and next_tok.kind == "IDENT"

    def _match_logic_op(self, kind: str) -> bool:
        if kind == "and":
            return self._match_keyword("and") or self._match_op("&&")
        if kind == "or":
            return self._match_keyword("or") or self._match_op("||")
        return False

    def _try_parse_attr_assignment(self) -> ast.SetAttrStmt | None:
        if not (
            self._is("IDENT")
            and self._peek(1).kind == "OP"
            and self._peek(1).lexeme == "."
            and self._peek(2).kind == "IDENT"
            and self._peek(3).kind == "OP"
            and self._peek(3).lexeme == "="
        ):
            return None
        ident = self._advance()
        target = ast.IdentifierExpr(name=ident.lexeme, span=self._span(ident))
        self._expect_op(".")
        attr = self._expect("IDENT", message="expected attribute name")
        assign_tok = self._expect_op("=")
        value = self._parse_expression()
        return ast.SetAttrStmt(target=target, attr=attr.lexeme, value=value, span=self._span(assign_tok))

    def _try_parse_index_assignment(self) -> ast.SetIndexStmt | None:
        if not (self._is("IDENT") and self._peek(1).kind == "OP" and self._peek(1).lexeme == "["):
            return None
        start = self.pos
        ident_tok = self._advance()
        target = ast.IdentifierExpr(name=ident_tok.lexeme, span=self._span(ident_tok))
        self._expect_op("[")
        index_expr = self._parse_expression()
        self._expect_op("]")
        if not self._match_op("="):
            self.pos = start
            return None
        assign_tok = self._prev()
        value = self._parse_expression()
        return ast.SetIndexStmt(target=target, index=index_expr, value=value, span=self._span(assign_tok))

    def _is_op(self, op: str) -> bool:
        tok = self._current()
        return tok.kind == "OP" and tok.lexeme == op

    def _match(self, kind: str) -> bool:
        if self._is(kind):
            self._advance()
            return True
        return False

    def _match_op(self, op: str) -> bool:
        if self._is_op(op):
            self._advance()
            return True
        return False

    def _match_keyword(self, text: str) -> bool:
        tok = self._current()
        if tok.kind == "KEYWORD" and tok.lexeme == text:
            self._advance()
            return True
        return False

    def _expect(self, kind: str, message: str) -> Token:
        if self._is(kind):
            return self._advance()
        self._error("E1004", message, self._current())
        return self._current()

    def _expect_op(self, op: str) -> Token:
        if self._is_op(op):
            return self._advance()
        self._error("E1005", f"expected '{op}'", self._current())
        return self._current()
