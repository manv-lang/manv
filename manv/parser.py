"""Recursive-descent parser for ManV source.

Why this file exists:
- Converts token streams into the AST consumed by semantics/interpreter/lowering.
- Centralizes grammar decisions so interpreter and compiled pipelines observe
  identical front-end syntax.
- Encodes migration compatibility (for example `class` alias for `type`) while
  preserving deterministic diagnostics.
"""

from __future__ import annotations

from . import ast
from .diagnostics import ManvError, Span, diag
from .lexer import Lexer
from .memory import NULL
from .tokens import Token

C_DECL_TYPES = {"int", "i32", "i64", "str", "array", "map", "u8", "usize", "float", "f32", "bool", "ptr", "void"}
MODULE_SEGMENT_KEYWORDS = {"str", "int", "float", "bool", "type", "array", "map"}


class Parser:
    def __init__(self, tokens: list[Token], file: str, source_lines: list[str]):
        self.tokens = tokens
        self.file = file
        self.source_lines = source_lines
        self.pos = 0
        self._after_block = False  # Set True after consuming a DEDENT (block end)
        self._no_cast = False  # True while parsing a `with` context, where `as` is not a cast

    def parse(self) -> ast.Program:
        declarations: list[object] = []
        statements: list[object] = []
        while not self._is("EOF"):
            self._consume_newlines()
            if self._is("EOF"):
                break
            decorators = self._parse_decorators()
            if self._match_keyword("fn"):
                declarations.append(self._parse_fn_decl(decorators=decorators))
            elif self._match_keyword("struct"):
                declarations.append(self._parse_struct_decl())
            elif self._match_keyword("type") or self._match_keyword("class"):
                if decorators:
                    self._error("E1012", "decorators are only supported on functions", self._current())
                declarations.append(self._parse_type_decl())
            elif self._match_keyword("impl"):
                if decorators:
                    self._error("E1012", "decorators are only supported on functions", self._current())
                declarations.append(self._parse_impl_decl())
            elif self._match_keyword("macro"):
                if decorators:
                    self._error("E1012", "decorators are only supported on functions", self._current())
                declarations.append(self._parse_macro_decl_stub())
            elif self._match_keyword("module"):
                if decorators:
                    self._error("E1012", "decorators are not supported on module declarations", self._current())
                declarations.append(self._parse_module_decl())
            elif self._match_keyword("extern"):
                if decorators:
                    self._error("E1012", "decorators are not supported on extern declarations", self._current())
                for node in self._parse_extern():
                    declarations.append(node)
                continue
            else:
                if decorators:
                    self._error("E1012", "decorators must appear immediately before a function declaration", self._current())
                result = self._parse_statement()
                if isinstance(result, list):
                    statements.extend(result)
                else:
                    statements.append(result)
        span = Span(self.file, 1, 1)
        docstring, statements = self._extract_leading_docstring(statements)
        return ast.Program(declarations=declarations, statements=statements, span=span, docstring=docstring)

    def _parse_fn_decl(self, *, decorators: list[ast.Decorator] | None = None) -> ast.FnDecl:
        fn_tok = self._prev()
        name_tok = self._expect("IDENT", message="expected function name")
        self._expect_op("(")
    
        params: list[ast.Param] = []
        if not self._is_op(")"):
            while True:
                params.append(self._parse_param(params))
                if self._match_op(","):
                    continue
                break
    
        self._expect_op(")")
        return_type = None
    
        if self._match_op("->"):
            return_type = self._parse_type(stop_ops={":"})
    
        self._expect_op(":")
        body = self._parse_block()
        docstring, body = self._extract_leading_docstring(body)
    
        return ast.FnDecl(
            name=name_tok.lexeme,
            params=params,
            return_type=return_type,
            body=body,
            span=self._span(fn_tok),
            decorators=list(decorators or []),
            docstring=docstring,
        )

    def _parse_struct_decl(self) -> ast.StructDecl:
        """`struct Name:` followed by an indented block of `T field` or `T field[N]` lines."""
        struct_tok = self._prev()
        name_tok = self._expect("IDENT", message="expected struct name")
        self._expect_op(":")
        self._consume_required_newline("expected newline after ':'")
        self._expect("INDENT", message="expected indented struct body")
        fields: list[ast.StructField] = []
        while not self._is("DEDENT") and not self._is("EOF"):
            self._consume_newlines()
            if self._is("DEDENT") or self._is("EOF"):
                break
            if not self._is_c_declaration_start():
                self._error("E1006", "expected a field declaration 'T name' inside struct", self._current())
            start = self._current()
            type_name = self._parse_decl_type()
            field_tok = self._expect("IDENT", message="expected field name")
            count = None
            if self._match_op("["):
                count = self._parse_expression()
                self._expect_op("]")
            if self._is_op("="):
                self._error("E1006", "struct fields cannot have initial values", self._current())
            fields.append(ast.StructField(name=field_tok.lexeme, type_name=type_name, count=count, span=self._span(start)))
            self._consume_required_newline("expected newline after struct field")
        self._expect("DEDENT", message="expected dedent")
        self._after_block = True
        return ast.StructDecl(name=name_tok.lexeme, fields=fields, span=self._span(struct_tok))

    def _parse_type_decl(self) -> ast.TypeDecl:
        type_tok = self._prev()
        name_tok = self._expect("IDENT", message="expected type name")
        base_names: list[str] = []
        if self._match_op("("):
            base_tok = self._expect("IDENT", message="expected base type name")
            base_names.append(base_tok.lexeme)
            while self._match_op(","):
                base_tok = self._expect("IDENT", message="expected base type name")
                base_names.append(base_tok.lexeme)
            self._expect_op(")")
        self._expect_op(":")
        docstring, attrs, methods = self._parse_type_block("type")
        return ast.TypeDecl(name=name_tok.lexeme, attrs=attrs, methods=methods, span=self._span(type_tok), base_names=base_names, docstring=docstring)

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
            decorators = self._parse_decorators()
            if not self._match_keyword("fn"):
                self._error("E1006", f"expected 'fn' inside {owner} block", self._current())
            methods.append(self._parse_fn_decl(decorators=decorators))
        self._expect("DEDENT", message="expected dedent")
        return methods

    def _parse_type_block(self, owner: str) -> tuple[str | None, list[ast.TypeAttrDecl], list[ast.FnDecl]]:
        """Parse a type body with optional docstring, immutable attrs, and methods."""

        self._consume_required_newline("expected newline after ':'")
        self._expect("INDENT", message="expected indented block")
        docstring: str | None = None
        attrs: list[ast.TypeAttrDecl] = []
        methods: list[ast.FnDecl] = []
        first_item = True
        while not self._is("DEDENT") and not self._is("EOF"):
            self._consume_newlines()
            if self._is("DEDENT") or self._is("EOF"):
                break
            if first_item and self._is("STRING"):
                doc_tok = self._advance()
                docstring = doc_tok.raw_lexeme if doc_tok.raw_lexeme is not None else doc_tok.lexeme
                self._consume_required_newline("expected newline after docstring")
                first_item = False
                continue

            decorators = self._parse_decorators()
            if self._match_keyword("fn"):
                methods.append(self._parse_fn_decl(decorators=decorators))
                first_item = False
                continue
            if decorators:
                self._error("E1012", "decorators are only supported on functions", self._current())
            if self._match_keyword("let"):
                self._error(
                    "E1015",
                    "'let' was removed: write 'int x = 0' or 'x = 0' for a field, 'const X = 0' for a constant",
                    self._prev(),
                )
            self._match_keyword("const")  # a const member is stored like a field; type attributes are immutable
            if self._is_c_declaration_start() or (self._is("IDENT") and self._peek(1).kind == "OP" and self._peek(1).lexeme == "="):
                attrs.append(self._parse_field_decl())
                self._consume_required_newline("expected newline after type attribute")
                first_item = False
                continue
            self._error("E1006", f"expected 'fn' or a field declaration inside {owner} block", self._current())
        self._expect("DEDENT", message="expected dedent")
        return docstring, attrs, methods

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

    def _parse_decorators(self) -> list[ast.Decorator]:
        decorators: list[ast.Decorator] = []
        while self._match_op("@"):
            decorators.append(self._parse_decorator())
            self._consume_required_newline("expected newline after decorator")
        return decorators

    def _parse_decorator(self) -> ast.Decorator:
        if self._is("IDENT") or self._is("KEYWORD"):
            name_tok = self._advance()
        else:
            self._error("E1004", "expected decorator name", self._current())
            name_tok = self._current()
        args: list[object] = []
        kwargs: dict[str, object] = {}
        if self._match_op("("):
            if not self._is_op(")"):
                while True:
                    if self._is("IDENT") and self._peek(1).kind == "OP" and self._peek(1).lexeme == "=":
                        key_tok = self._advance()
                        self._expect_op("=")
                        kwargs[key_tok.lexeme] = self._parse_expression()
                    else:
                        args.append(self._parse_expression())
                    if self._match_op(","):
                        continue
                    break
            self._expect_op(")")
        return ast.Decorator(name=name_tok.lexeme, args=args, kwargs=kwargs, span=self._span(name_tok))

    def _parse_statement(self) -> object:
        if self._match_keyword("fn"):
            # Local function declaration — defines a named function in current scope.
            decl = self._parse_fn_decl()
            return decl
        if self._match_keyword("extern"):
            return list(self._parse_extern())
        if self._match_keyword("include"):
            result = self._parse_include_stmt()
            self._consume_required_newline("expected newline after include")
            return result
        if self._is_c_declaration_start():
            stmt = self._parse_c_decl_stmt()
            self._consume_required_newline("expected newline after declaration")
            return stmt
        if self._match_keyword("let"):
            self._error(
                "E1015",
                "'let' was removed: write 'x = ...' or 'T x = ...'; use 'const' for constants",
                self._prev(),
            )
        if self._match_keyword("const"):
            stmt = self._parse_const_stmt()
            self._consume_required_newline("expected newline after const declaration")
            return stmt
        if self._match_keyword("return"):
            stmt = self._parse_return_stmt()
            self._consume_required_newline("expected newline after return")
            return stmt
        if self._match_keyword("raise"):
            stmt = self._parse_raise_stmt()
            self._consume_required_newline("expected newline after raise")
            return stmt
        if self._match_keyword("syscall"):
            stmt = self._parse_syscall_stmt()
            self._consume_required_newline("expected newline after syscall")
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
        if self._match_keyword("for"):
            return self._parse_for_stmt()
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

        if self._is_op("*"):
            return self._parse_deref_assignment()

        if self._is("IDENT") and self._peek(1).kind == "OP" and self._peek(1).lexeme in {"+=", "-=", "*=", "/=", "%="}:
            name_tok = self._advance()
            op_tok = self._advance()
            value = self._parse_expression()
            self._consume_required_newline("expected newline after augmented assignment")
            return ast.AugAssignStmt(name=name_tok.lexeme, op=op_tok.lexeme, value=value, span=self._span(op_tok))

        if self._is("IDENT") and self._peek(1).kind == "OP" and self._peek(1).lexeme == "=":
            name_tok = self._advance()
            assign_tok = self._advance()
            value = self._parse_expression_or_tuple()
            self._consume_required_newline("expected newline after assignment")
            return ast.AssignStmt(name=name_tok.lexeme, value=value, span=self._span(assign_tok))

        if self._is_destructure_start():
            first = self._current()
            targets = [ast.AssignStmt(name=self._advance().lexeme, value=None, span=self._span(first))]
            while self._match_op(","):
                tok = self._expect("IDENT", message="expected a name to unpack into")
                targets.append(ast.AssignStmt(name=tok.lexeme, value=None, span=self._span(tok)))
            assign_tok = self._expect_op("=")
            value = self._parse_expression_or_tuple()
            self._consume_required_newline("expected newline after assignment")
            return ast.DestructureStmt(targets=targets, value=value, span=self._span(assign_tok))

        attr_assign = self._try_parse_attr_assignment()
        if attr_assign is not None:
            self._consume_required_newline("expected newline after attribute assignment")
            return attr_assign

        aug_attr = self._try_parse_aug_attr_assignment()
        if aug_attr is not None:
            self._consume_required_newline("expected newline after augmented attribute assignment")
            return aug_attr

        index_assign = self._try_parse_index_assignment()
        if index_assign is not None:
            self._consume_required_newline("expected newline after index assignment")
            return index_assign

        aug_index = self._try_parse_aug_index_assignment()
        if aug_index is not None:
            self._consume_required_newline("expected newline after augmented index assignment")
            return aug_index

        if self._match_keyword("with"):
            return self._parse_with_stmt()

        if self._match_keyword("match"):
            return self._parse_match_stmt()

        expr = self._parse_expression()
        assignment = self._assignment_to(expr)
        if assignment is not None:
            self._consume_required_newline("expected newline after assignment")
            return assignment
        self._consume_required_newline("expected newline after expression")
        return ast.ExprStmt(expr=expr, span=self._span_from_expr(expr))

    def _parse_deref_assignment(self) -> object:
        """`*p = v` and `*p += v` (a statement starting with `*`)."""
        star = self._current()
        target = self._parse_unary()
        assignment = self._assignment_to(target)
        if assignment is None:
            self._error("E1002", "a statement starting with '*' must assign through the pointer: '*p = value'", star)
        self._consume_required_newline("expected newline after assignment")
        return assignment

    def _assignment_to(self, target: object) -> object | None:
        """If `=` or `op=` follows `target`, parse the right side and build the matching store statement."""
        aug = {"+=": "+", "-=": "-", "*=": "*", "/=": "/", "%=": "%"}
        tok = self._current()
        if tok.kind != "OP" or not (tok.lexeme == "=" or tok.lexeme in aug):
            return None
        if not isinstance(target, (ast.AttributeExpr, ast.IndexExpr, ast.DerefExpr)):
            return None
        self._advance()
        value = self._parse_expression()
        op = aug.get(tok.lexeme)
        span = self._span(tok)
        if isinstance(target, ast.DerefExpr):
            return ast.SetDerefStmt(target=target.operand, value=value, span=span, op=op)
        if isinstance(target, ast.AttributeExpr):
            if op is None:
                return ast.SetAttrStmt(target=target.value, attr=target.attr, value=value, span=span)
            return ast.AugAssignAttrStmt(target=target.value, attr=target.attr, op=tok.lexeme, value=value, span=span)
        if op is None:
            return ast.SetIndexStmt(target=target.value, index=target.index, value=value, span=span)
        return ast.AugAssignIndexStmt(target=target.value, index=target.index, op=tok.lexeme, value=value, span=span)

    def _parse_param(self, previous: list[ast.Param]) -> ast.Param:
        """`name`, `name: T`, `name = default` or `name: T = default`. Once one parameter has a
        default, every later one needs a default too."""
        p_tok = self._expect("IDENT", message="expected parameter name")
        p_type = None
        if self._match_op(":"):
            p_type = self._parse_type(stop_ops={",", ")", "="})
        default = None
        if self._match_op("="):
            default = self._parse_expression()
        elif previous and previous[-1].default is not None:
            self._error("E1019", f"parameter '{p_tok.lexeme}' needs a default value: it follows a parameter that has one", p_tok)
        return ast.Param(name=p_tok.lexeme, type_name=p_type, span=self._span(p_tok), default=default)

    def _is_destructure_start(self) -> bool:
        """`a, b = ...`: names separated by commas, then `=`."""
        i = 0
        while self._peek(i).kind == "IDENT" and self._peek(i + 1).kind == "OP" and self._peek(i + 1).lexeme == ",":
            i += 2
        if i == 0:
            return False
        return self._peek(i).kind == "IDENT" and self._peek(i + 1).kind == "OP" and self._peek(i + 1).lexeme == "="

    def _parse_expression_or_tuple(self) -> object:
        """An expression, or `a, b, c` (a tuple without parentheses) where a statement allows it."""
        first = self._parse_expression()
        if not self._is_op(","):
            return first
        elements = [first]
        while self._match_op(","):
            if self._is("NEWLINE"):
                break
            elements.append(self._parse_expression())
        return ast.TupleExpr(elements=elements, span=self._span_from_expr(first))

    def _parse_const_stmt(self) -> ast.LetStmt:
        """`const NAME = value` or `const T NAME = value`; the value is required."""
        const_tok = self._prev()
        type_name = self._parse_decl_type() if self._is_c_declaration_start() else None
        name_tok = self._expect("IDENT", message="expected constant name")
        if not self._is_op("="):
            self._error("E1016", f"constant '{name_tok.lexeme}' needs a value: 'const {name_tok.lexeme} = ...'", name_tok)
        self._advance()
        value = self._parse_expression()
        return ast.LetStmt(
            name=name_tok.lexeme,
            type_name=type_name,
            value=value,
            span=self._span(const_tok),
            is_const=True,
        )

    def _parse_field_decl(self) -> ast.TypeAttrDecl:
        """A field or constant in a type body: `T name = value` or `name = value` (after any `const`)."""
        start = self._current()
        type_name = self._parse_decl_type() if self._is_c_declaration_start() else None
        name_tok = self._expect("IDENT", message="expected attribute name")
        self._expect_op("=")
        value = self._parse_expression()
        return ast.TypeAttrDecl(name=name_tok.lexeme, type_name=type_name, value=value, span=self._span(start))

    def _parse_decl_type(self) -> str:
        """Consume the type in `T name`: a type word, optionally `const`, `*`s (`int**`) and `[]` pairs (`f32[]`)."""
        type_name = self._advance().lexeme
        if self._is("KEYWORD") and self._current().lexeme == "const" and self._peek(1).lexeme == "*":
            self._advance()
            type_name = "const " + type_name
        while self._is_op("*"):
            self._advance()
            type_name += "*"
        while self._is_op("[") and self._peek(1).kind == "OP" and self._peek(1).lexeme == "]":
            self._advance()
            self._advance()
            type_name += "[]"
        return type_name

    def _parse_c_decl_stmt(self) -> ast.LetStmt:
        type_tok = self._current()
        type_name = self._parse_decl_type()
        name_tok = self._expect("IDENT", message="expected variable name")
        array_size: object | None = None

        c_array = False
        if self._match_op("["):
            array_size = self._parse_expression()
            self._expect_op("]")
            if type_name != "array":
                c_array = True  # `array xs[3]` is the managed array; any other element type is C memory
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
            c_array=c_array,
        )

    def _parse_return_stmt(self) -> ast.ReturnStmt:
        tok = self._prev()
        if self._is("NEWLINE"):
            return ast.ReturnStmt(value=None, span=self._span(tok))
        value = self._parse_expression_or_tuple()
        return ast.ReturnStmt(value=value, span=self._span(tok))

    def _parse_raise_stmt(self) -> ast.RaiseStmt:
        tok = self._prev()
        if self._is("NEWLINE"):
            return ast.RaiseStmt(value=None, span=self._span(tok))
        value = self._parse_expression()
        return ast.RaiseStmt(value=value, span=self._span(tok))

    def _parse_syscall_stmt(self) -> ast.SyscallStmt:
        expr = self._parse_syscall_expr()
        return ast.SyscallStmt(target=expr.target, args=expr.args, span=expr.span)

    def _parse_syscall_expr(self) -> ast.SyscallExpr:
        tok = self._prev()
        self._expect_op("(")
        target = self._parse_expression()
        args: list[object] = []
        while self._match_op(","):
            args.append(self._parse_expression())
        self._expect_op(")")
        return ast.SyscallExpr(target=target, args=args, span=self._span(tok))

    def _parse_extern(self) -> list[ast.ExternBlock]:
        """Parse `extern "lib" fn ...` or `extern "lib": block` forms."""
        extern_tok = self._prev()
        if not self._is("STRING"):
            self._error("E1013", "expected library name string after 'extern'", self._current())
        lib_tok = self._advance()
        lib = lib_tok.lexeme  # the processed string value from lexer

        span = self._span(extern_tok)

        # Single-fn form: extern "lib" fn name(...)
        if self._match_keyword("fn"):
            fn_decl = self._parse_extern_fn(lib)
            block = ast.ExternBlock(lib=lib, fns=[fn_decl], span=span)
            self._consume_newlines()
            return [block]

        # Block form: extern "lib":\n    INDENT fn... fn... DEDENT
        self._expect_op(":")
        self._consume_required_newline("expected newline after 'extern' block header")
        self._expect("INDENT", message="expected indented extern block")
        fns: list[ast.ExternFnDecl] = []
        while not self._is("DEDENT") and not self._is("EOF"):
            self._consume_newlines()
            if self._is("DEDENT") or self._is("EOF"):
                break
            if not self._match_keyword("fn"):
                self._error("E1006", "expected 'fn' inside extern block", self._current())
            fns.append(self._parse_extern_fn(lib))
        self._expect("DEDENT", message="expected dedent after extern block")
        block = ast.ExternBlock(lib=lib, fns=fns, span=span)
        return [block]

    def _parse_extern_fn(self, lib: str) -> ast.ExternFnDecl:
        """Parse a single extern fn declaration (after `fn` keyword consumed)."""
        fn_tok = self._prev()
        name_tok = self._expect("IDENT", message="expected extern function name")
        self._expect_op("(")

        params: list[ast.Param] = []
        varargs = False
        if not self._is_op(")"):
            while True:
                # Handle `...` varargs marker (tokenized as `..` + `.`)
                if self._is_op("..") and self._peek(1).kind == "OP" and self._peek(1).lexeme == ".":
                    self._advance()  # consume ".."
                    self._advance()  # consume "."
                    varargs = True
                    break
                p_tok = self._expect("IDENT", message="expected parameter name")
                p_type = None
                if self._match_op(":"):
                    p_type = self._parse_type(stop_ops={",", ")"})
                params.append(ast.Param(name=p_tok.lexeme, type_name=p_type, span=self._span(p_tok)))
                if self._match_op(","):
                    # Check for trailing `...` varargs
                    if self._is_op("..") and self._peek(1).kind == "OP" and self._peek(1).lexeme == ".":
                        self._advance()  # consume ".."
                        self._advance()  # consume "."
                        varargs = True
                        break
                    continue
                break

        self._expect_op(")")

        return_type: str | None = None
        if self._match_op("->"):
            return_type = self._parse_type(stop_ops={",", ")"}, stop_keywords={"as"})

        # Optional `as "c_symbol_name"` alias
        c_name = name_tok.lexeme
        if self._match_keyword("as"):
            if not self._is("STRING"):
                self._error("E1013", "expected string after 'as' in extern fn alias", self._current())
            alias_tok = self._advance()
            c_name = alias_tok.lexeme

        self._consume_required_newline("expected newline after extern fn declaration")

        return ast.ExternFnDecl(
            name=name_tok.lexeme,
            c_name=c_name,
            lib=lib,
            params=params,
            return_type=return_type if return_type != "void" else None,
            varargs=varargs,
            span=self._span(fn_tok),
        )

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

    def _parse_module_decl(self) -> ast.ModuleDecl:
        """Parse ``module <name>`` declaration at top of file.

        Accepts dotted names: ``module pkg.sub.leaf``.
        """
        tok = self._prev()  # 'module' keyword already consumed
        name_tok = self._expect("IDENT", message="expected module name after 'module'")
        parts = [name_tok.lexeme]
        while self._match_op("."):
            seg = self._expect("IDENT", message="expected module name segment after '.'")
            parts.append(seg.lexeme)
        self._consume_required_newline("expected newline after module declaration")
        return ast.ModuleDecl(name=".".join(parts), span=self._span(tok))

    def _parse_include_stmt(self) -> ast.IncludeStmt | list[ast.IncludeStmt]:
        """Parse ``include`` statement in all supported forms.

        Forms parsed:
        - ``include mymod``                      whole-module, bound as ``mymod``
        - ``include mymod as m``                 whole-module, bound as ``m``
        - ``include pkg.sub``                    whole-module, bound as ``sub``
        - ``include Foo from mymod``             single name
        - ``include Foo from mymod as F``        single name with alias
        - ``include A, B from mymod``            multiple names → list[IncludeStmt]
        - ``include A as X, B as Y from mymod`` multiple names with per-name aliases
        - ``include Foo from .sibling``          relative single name

        Disambiguation rule: after leading dots + first IDENT, if the next token
        is ``,`` or keyword ``from`` → name-include form; otherwise → whole-module form.
        """
        tok = self._prev()  # 'include' keyword already consumed

        # Leading dots encode relative depth (same convention as old import).
        level = 0
        while self._match_op(".."):
            level += 2
        while self._match_op("."):
            level += 1

        first_tok = self._expect_module_name("expected name or module path after 'include'")
        first = first_tok.lexeme

        # ── Disambiguation ────────────────────────────────────────────────────
        # We need to decide between the name-include and whole-module forms.
        # The tricky case is `include A as X from mymod` where `as` appears
        # before `from`; a simple single-token lookahead would miss this.
        #
        # Rules after reading the first IDENT:
        #   `,`  after first → name-include (multiple names)
        #   `.`  after first → whole-module (dotted path)
        #   `as` after first → tentatively read alias, then re-discriminate:
        #       `,` or `from` next → name-include (A as X [, B ...] from mod)
        #       otherwise          → whole-module (mod as alias)
        #   `from` after first → name-include (no alias on first name)
        #   anything else      → whole-module (bare module name)

        if self._current_is_op(",") or self._current_is_keyword("from"):
            # Unambiguous name-include.
            names: list[tuple[str, str | None]] = [(first, None)]
            while self._match_op(","):
                name_tok2 = self._expect("IDENT", message="expected symbol name")
                alias2: str | None = None
                if self._match_keyword("as"):
                    alias_tok3 = self._expect("IDENT", message="expected alias after 'as'")
                    alias2 = alias_tok3.lexeme
                names.append((name_tok2.lexeme, alias2))
            if not self._match_keyword("from"):
                self._error("E1004", "expected 'from' in include statement", self._current())
            mod_path, mod_level = self._parse_module_path(allow_empty=True)
            if level == 0:
                level = mod_level
            if len(names) == 1:
                name_s, alias_s = names[0]
                return ast.IncludeStmt(
                    module=mod_path, name=name_s, alias=alias_s,
                    span=self._span(tok), level=level,
                )
            return [
                ast.IncludeStmt(module=mod_path, name=n, alias=a,
                                span=self._span(tok), level=level)
                for n, a in names
            ]

        if self._current_is_keyword("as"):
            # Could be either form — consume the alias token then check what follows.
            self._advance()  # consume 'as'
            alias_tok2 = self._expect("IDENT", message="expected alias after 'as'")
            tentative_alias = alias_tok2.lexeme

            if self._current_is_op(",") or self._current_is_keyword("from"):
                # name-include: include A as X [, B as Y ...] from mod
                names2: list[tuple[str, str | None]] = [(first, tentative_alias)]
                while self._match_op(","):
                    name_tok3 = self._expect("IDENT", message="expected symbol name")
                    alias3: str | None = None
                    if self._match_keyword("as"):
                        alias_tok4 = self._expect("IDENT", message="expected alias after 'as'")
                        alias3 = alias_tok4.lexeme
                    names2.append((name_tok3.lexeme, alias3))
                if not self._match_keyword("from"):
                    self._error("E1004", "expected 'from' in include statement", self._current())
                mod_path2, mod_level2 = self._parse_module_path(allow_empty=True)
                if level == 0:
                    level = mod_level2
                if len(names2) == 1:
                    name_s2, alias_s2 = names2[0]
                    return ast.IncludeStmt(
                        module=mod_path2, name=name_s2, alias=alias_s2,
                        span=self._span(tok), level=level,
                    )
                return [
                    ast.IncludeStmt(module=mod_path2, name=n, alias=a,
                                    span=self._span(tok), level=level)
                    for n, a in names2
                ]
            else:
                # whole-module: include mymod as alias
                return ast.IncludeStmt(
                    module=first, name=None, alias=tentative_alias,
                    span=self._span(tok), level=level,
                )

        # ── whole-module form: include pkg[.sub] [as alias] ──
        parts = [first]
        while self._match_op("."):
            if not self._is_module_segment_token(self._current()):
                self._error("E1004", "expected module path segment", self._current())
            parts.append(self._advance().lexeme)
        alias_wm: str | None = None
        if self._match_keyword("as"):
            alias_tok_wm = self._expect("IDENT", message="expected alias after 'as'")
            alias_wm = alias_tok_wm.lexeme
        return ast.IncludeStmt(
            module=".".join(parts), name=None, alias=alias_wm,
            span=self._span(tok), level=level,
        )

    def _current_is_op(self, op: str) -> bool:
        """Return True if the current token is an operator matching *op*."""
        t = self._current()
        return t.kind == "OP" and t.lexeme == op

    def _current_is_keyword(self, kw: str) -> bool:
        """Return True if the current token is a keyword matching *kw*."""
        t = self._current()
        return t.kind == "KEYWORD" and t.lexeme == kw

    def _expect_module_name(self, message: str) -> Token:
        """Accept an IDENT or a type-keyword that is valid as a module name.

        Type keywords such as ``str``, ``int``, ``float`` are valid module
        names (e.g. ``include str as s``).  A plain ``_expect("IDENT", ...)``
        would reject them because the lexer classifies them as KEYWORD tokens.
        """
        tok = self._current()
        if tok.kind == "IDENT" or self._is_module_segment_token(tok):
            return self._advance()
        self._error("E1004", message, tok)
        raise AssertionError("unreachable")

    def _parse_module_path(self, *, allow_empty: bool) -> tuple[str, int]:
        # Leading dots encode relative depth.
        level = 0
        while True:
            if self._match_op(".."):
                # The lexer tokenizes `..` as one operator for range syntax.
                # Relative imports reuse the same token stream, so the parser
                # must count that token as two leading package hops here.
                level += 2
                continue
            if self._match_op("."):
                level += 1
                continue
            break

        # Remaining identifier chain is the explicit module tail.
        parts: list[str] = []
        if self._is_module_segment_token(self._current()):
            parts.append(self._advance().lexeme)
            while self._match_op("."):
                if not self._is_module_segment_token(self._current()):
                    self._error("E1004", "expected module path segment", self._current())
                parts.append(self._advance().lexeme)
        elif level == 0 or not allow_empty:
            self._error("E1004", "expected module path", self._current())
        return ".".join(parts), level

    def _is_module_segment_token(self, tok: Token) -> bool:
        if tok.kind == "IDENT":
            return True
        return tok.kind == "KEYWORD" and tok.lexeme in MODULE_SEGMENT_KEYWORDS

    def _parse_if_stmt(self) -> ast.IfStmt:
        if_tok = self._prev()
        condition = self._parse_expression()
        self._expect_op(":")
        then_body = self._parse_block()
        else_body: list[object] = []
        if self._is("KEYWORD") and self._current().lexeme == "elif":
            # `elif` is an `else` whose body is one nested `if`.
            self._advance()
            else_body = [self._parse_if_stmt()]
        elif self._match_keyword("else"):
            self._expect_op(":")
            else_body = self._parse_block()
        return ast.IfStmt(condition=condition, then_body=then_body, else_body=else_body, span=self._span(if_tok))

    def _parse_while_stmt(self) -> ast.WhileStmt:
        while_tok = self._prev()
        condition = self._parse_expression()
        self._expect_op(":")
        body = self._parse_block()
        else_body: list[object] = []
        if self._match_keyword("else"):
            self._expect_op(":")
            else_body = self._parse_block()
        return ast.WhileStmt(condition=condition, body=body, span=self._span(while_tok), else_body=else_body)

    def _parse_for_stmt(self) -> ast.ForStmt:
        for_tok = self._prev()
        name_tok = self._expect("IDENT", message="expected loop variable name")
        unpack: list[Token] = []
        if self._is_op(","):
            unpack = [name_tok]
            while self._match_op(","):
                unpack.append(self._expect("IDENT", message="expected loop variable name"))
        if not self._match_keyword("in"):
            self._error("E1013", "expected 'in' after loop variable", self._current())
        iterable = self._parse_expression()
        self._expect_op(":")
        body = self._parse_block()
        if unpack:
            hidden = f"__for_{for_tok.line}_{for_tok.column}"
            unpacking = ast.DestructureStmt(
                targets=[ast.AssignStmt(name=t.lexeme, value=None, span=self._span(t)) for t in unpack],
                value=ast.IdentifierExpr(name=hidden, span=self._span(name_tok)),
                span=self._span(name_tok),
                always_declare=True,
            )
            body = [unpacking, *body]
            name_tok = Token(kind="IDENT", lexeme=hidden, line=name_tok.line, column=name_tok.column)
        else_body: list[object] = []
        if self._match_keyword("else"):
            self._expect_op(":")
            else_body = self._parse_block()
        return ast.ForStmt(var_name=name_tok.lexeme, iterable=iterable, body=body, span=self._span(for_tok), else_body=else_body)

    def _parse_block(self) -> list[object]:
        self._consume_required_newline("expected newline after ':'")
        self._expect("INDENT", message="expected indented block")
        body: list[object] = []
        while not self._is("DEDENT") and not self._is("EOF"):
            self._consume_newlines()
            if self._is("DEDENT") or self._is("EOF"):
                break
            result = self._parse_statement()
            if isinstance(result, list):
                body.extend(result)
            else:
                body.append(result)
        self._expect("DEDENT", message="expected dedent")
        self._after_block = True
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

    def _parse_type(self, stop_ops: set[str], stop_keywords: set[str] | None = None) -> str:
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
            if tok.kind == "KEYWORD" and bracket_depth == 0 and stop_keywords and tok.lexeme in stop_keywords:
                break
            parts.append(self._advance().lexeme)
            if parts[-1] == "const":
                parts.append(" ")
        result = "".join(parts).strip()
        if not result:
            tok = self._current()
            self._error("E1001", "expected type annotation", tok)
        return result

    def _parse_expression(self) -> object:
        return self._parse_conditional()

    def _parse_conditional(self) -> object:
        """`a if cond else b` (right-associative, the loosest operator)."""
        expr = self._parse_range()
        if self._is("KEYWORD") and self._current().lexeme == "if":
            self._advance()
            condition = self._parse_or()
            if not self._match_keyword("else"):
                self._error("E1017", "expected 'else' in conditional expression", self._current())
            other = self._parse_conditional()
            return ast.CondExpr(condition=condition, then_expr=expr, else_expr=other, span=self._span_from_expr(expr))
        return expr

    def _parse_range(self) -> object:
        expr = self._parse_or()
        if self._match_op(".."):
            right = None if self._is_op("]") else self._parse_or()
            return ast.RangeExpr(start=expr, stop=right, span=self._span_from_expr(expr))
        return expr

    def _parse_or(self) -> object:
        expr = self._parse_and()
        while self._match_logic_op("or"):
            op = self._prev().lexeme
            right = self._parse_and()
            expr = ast.BinaryExpr(left=expr, op=op, right=right, span=self._span_from_expr(expr))
        return expr

    def _parse_and(self) -> object:
        expr = self._parse_not()
        while self._match_logic_op("and"):
            op = self._prev().lexeme
            right = self._parse_not()
            expr = ast.BinaryExpr(left=expr, op=op, right=right, span=self._span_from_expr(expr))
        return expr

    def _parse_not(self) -> object:
        """`not x` is looser than comparison: `not a in b` is `not (a in b)`."""
        if self._match_keyword("not"):
            op_tok = self._prev()
            return ast.UnaryExpr(op="not", expr=self._parse_not(), span=self._span(op_tok))
        return self._parse_equality()

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
            elif self._match_keyword("in"):
                op = "in"
            elif self._is("KEYWORD") and self._current().lexeme == "not" and self._peek(1).lexeme == "in":
                self._advance()
                self._advance()
                right = self._parse_term()
                inner = ast.BinaryExpr(left=expr, op="in", right=right, span=self._span_from_expr(expr))
                expr = ast.UnaryExpr(op="not", expr=inner, span=self._span_from_expr(expr))
                continue
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
        """A prefix expression followed by any number of `as T` casts (`-x as int` is `(-x) as int`)."""
        expr = self._parse_prefix()
        while not self._no_cast and self._is("KEYWORD") and self._current().lexeme == "as":
            as_tok = self._advance()
            expr = ast.CastExpr(operand=expr, type_name=self._parse_cast_type(), span=self._span(as_tok))
        return expr

    def _parse_prefix(self) -> object:
        if self._match_op("-"):
            op_tok = self._prev()
            right = self._parse_prefix()
            return ast.UnaryExpr(op="-", expr=right, span=self._span(op_tok))
        if self._match_op("!"):
            op_tok = self._prev()
            right = self._parse_prefix()
            return ast.UnaryExpr(op="!", expr=right, span=self._span(op_tok))
        if self._match_op("&"):
            op_tok = self._prev()
            return ast.AddrOfExpr(operand=self._parse_prefix(), span=self._span(op_tok))
        if self._match_op("*"):
            op_tok = self._prev()
            return ast.DerefExpr(operand=self._parse_prefix(), span=self._span(op_tok))
        return self._parse_postfix()

    _CAST_TYPE_WORDS = C_DECL_TYPES | {"void", "u16", "u32", "u64", "i8", "i16", "f64"}

    def _parse_cast_type(self) -> str:
        """The type after `as`: `usize`, `u8*`, `const char*`, `void**`."""
        const = False
        if self._is("KEYWORD") and self._current().lexeme == "const":
            self._advance()
            const = True
        tok = self._current()
        if not (tok.kind == "IDENT" or (tok.kind == "KEYWORD" and tok.lexeme in self._CAST_TYPE_WORDS)):
            self._error("E1021", "expected a type after 'as'", tok)
        self._advance()
        name = tok.lexeme
        if const:
            name = "const " + name
        while self._is_op("*") and self._star_is_pointer():
            self._advance()
            name += "*"
        return name

    def _star_is_pointer(self) -> bool:
        """In `p as T * ...`: is this `*` part of the type, or a multiplication? It is part of the
        type when nothing that could start an operand follows (`)`, `,`, newline, another `*`, ...)."""
        nxt = self._peek(1)
        if nxt.kind in {"NEWLINE", "EOF"}:
            return True
        if nxt.kind == "OP":
            return nxt.lexeme in {")", "]", "}", ",", ":", "=", "==", "!=", "<", "<=", ">", ">=", "+", "*", "..", "->", "&&", "||"}
        if nxt.kind == "KEYWORD":
            return nxt.lexeme in {"and", "or", "if", "else", "in", "as", "for"}
        return False

    def _parse_postfix(self) -> object:
        expr = self._parse_primary()
        # Macro call: IdentifierExpr followed by !( ... )
        if (
            isinstance(expr, ast.IdentifierExpr)
            and self._is_op("!")
            and self._peek(1).kind == "OP"
            and self._peek(1).lexeme == "("
        ):
            self._advance()  # consume !
            self._advance()  # consume (
            macro_args: list[object] = []
            if not self._is_op(")"):
                while True:
                    macro_args.append(self._parse_expression())
                    if self._match_op(","):
                        continue
                    break
            close = self._expect_op(")")
            expr = ast.MacroCallExpr(name=expr.name, args=macro_args, span=self._span(close))
        while True:
            if self._match_op("("):
                args: list[object] = []
                kwargs: list[tuple[str, object]] = []
                if not self._is_op(")"):
                    while True:
                        if self._is("IDENT") and self._peek(1).kind == "OP" and self._peek(1).lexeme == "=":
                            name_tok = self._advance()
                            self._advance()
                            if any(name == name_tok.lexeme for name, _ in kwargs):
                                self._error("E1020", f"keyword argument '{name_tok.lexeme}' is given twice", name_tok)
                            kwargs.append((name_tok.lexeme, self._parse_expression()))
                        else:
                            if kwargs:
                                self._error("E1020", "a positional argument cannot follow a keyword argument", self._current())
                            args.append(self._parse_expression())
                        if self._match_op(","):
                            if self._is_op(")"):
                                break
                            continue
                        break
                close = self._expect_op(")")
                expr = ast.CallExpr(callee=expr, args=args, span=self._span(close), kwargs=kwargs)
                continue
            if self._match_op("["):
                idx = self._parse_index()
                close = self._expect_op("]")
                expr = ast.IndexExpr(value=expr, index=idx, span=self._span(close))
                continue
            if self._match_op("."):
                attr = self._expect_attr_name()
                expr = ast.AttributeExpr(value=expr, attr=attr.lexeme, span=self._span(attr))
                continue
            if self._is_op("->"):
                arrow = self._advance()
                attr = self._expect_attr_name()
                deref = ast.DerefExpr(operand=expr, span=self._span(arrow))
                expr = ast.AttributeExpr(value=deref, attr=attr.lexeme, span=self._span(attr))
                continue
            break
        return expr

    def _parse_index(self) -> object:
        """The inside of `[...]`: an expression or a slice `a..b`; either bound may be left out."""
        if self._is_op(".."):
            tok = self._advance()
            stop = None if self._is_op("]") else self._parse_or()
            return ast.RangeExpr(start=None, stop=stop, span=self._span(tok))
        return self._parse_expression()

    def _parse_primary(self) -> object:
        tok = self._current()
        if self._match("NUMBER"):
            if "." in tok.lexeme or "e" in tok.lexeme:  # the lexer normalizes: floats have '.' or an exponent
                return ast.LiteralExpr(value=float(tok.lexeme), literal_type="float", span=self._span(tok))
            return ast.LiteralExpr(value=int(tok.lexeme), literal_type="int", span=self._span(tok))
        if self._match("STRING"):
            return ast.LiteralExpr(value=tok.lexeme, literal_type="str", span=self._span(tok), raw_value=tok.raw_lexeme)
        if self._match("FSTRING"):
            return self._parse_fstring(tok)
        if self._match_keyword("true") or self._match_keyword("True"):
            return ast.LiteralExpr(value=True, literal_type="bool", span=self._span(tok))
        if self._match_keyword("false") or self._match_keyword("False"):
            return ast.LiteralExpr(value=False, literal_type="bool", span=self._span(tok))
        if self._match_keyword("none"):
            return ast.LiteralExpr(value=None, literal_type="none", span=self._span(tok))
        if self._match_keyword("null"):
            return ast.LiteralExpr(value=NULL, literal_type="null", span=self._span(tok))
        if self._match_keyword("sizeof"):
            return self._parse_sizeof(tok)
        if self._match_keyword("syscall"):
            return self._parse_syscall_expr()
        if tok.kind == "KEYWORD" and tok.lexeme in {"type", "int", "i32", "i64", "float", "f32", "bool", "str", "void", "ptr"}:
            self._advance()
            return ast.IdentifierExpr(name=tok.lexeme, span=self._span(tok))
        if self._match("IDENT"):
            return ast.IdentifierExpr(name=tok.lexeme, span=self._span(tok))
        if self._match_op("("):
            if self._is_op(")"):
                close = self._advance()
                return ast.TupleExpr(elements=[], span=self._span(close))
            expr = self._parse_expression()
            if self._is_op(","):
                elements = [expr]
                while self._match_op(","):
                    if self._is_op(")"):
                        break
                    elements.append(self._parse_expression())
                close = self._expect_op(")")
                return ast.TupleExpr(elements=elements, span=self._span(close))
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
        if self._match_keyword("fn"):
            return self._parse_lambda_expr()
        self._error("E1002", "expected expression", tok)
        return ast.LiteralExpr(value=None, literal_type="none", span=self._span(tok))

    def _parse_fstring(self, tok: Token) -> object:
        """Split `f"..."` into literal text and `{expression[:spec]}` fields. `{{` and `}}` are literal braces."""
        text = tok.lexeme
        parts: list[object] = []
        buf: list[str] = []
        i = 0
        while i < len(text):
            c = text[i]
            if c == "}":
                if text[i + 1 : i + 2] != "}":
                    self._error("E1018", "single '}' in f-string (write '}}' for a literal brace)", tok)
                buf.append("}")
                i += 2
                continue
            if c != "{":
                buf.append(c)
                i += 1
                continue
            if text[i + 1 : i + 2] == "{":
                buf.append("{")
                i += 2
                continue
            end, colon = self._scan_fstring_field(text, i + 1, tok)
            expr_text = text[i + 1 : colon if colon >= 0 else end].strip()
            spec = text[colon + 1 : end] if colon >= 0 else ""
            if not expr_text:
                self._error("E1018", "empty expression in f-string", tok)
            if buf:
                parts.append("".join(buf))
                buf = []
            parts.append((self._parse_fstring_expr(expr_text, tok), spec))
            i = end + 1
        if buf:
            parts.append("".join(buf))
        if all(isinstance(p, str) for p in parts):
            return ast.LiteralExpr(value="".join(parts), literal_type="str", span=self._span(tok))
        return ast.FStringExpr(parts=parts, span=self._span(tok))

    def _scan_fstring_field(self, text: str, start: int, tok: Token) -> tuple[int, int]:
        """Return (index of the closing '}', index of the top-level ':' or -1) for a field starting at `start`."""
        depth = 0
        quote: str | None = None
        colon = -1
        j = start
        while j < len(text):
            d = text[j]
            if quote is not None:
                if d == quote:
                    quote = None
            elif d in {'"', "'"}:
                quote = d
            elif d in "([{":
                depth += 1
            elif d in ")]":
                depth -= 1
            elif d == "}":
                if depth == 0:
                    return j, colon
                depth -= 1
            elif d == ":" and depth == 0 and colon < 0:
                colon = j
            j += 1
        self._error("E1018", "unterminated '{' in f-string", tok)
        return -1, -1

    def _parse_fstring_expr(self, expr_text: str, tok: Token) -> object:
        # Pad with newlines so errors and runtime failures report the f-string's own line.
        source = "\n" * (tok.line - 1) + expr_text
        sub = Parser(Lexer(source, self.file).tokenize(), self.file, source.splitlines())
        sub._consume_newlines()
        expr = sub._parse_expression()
        sub._consume_newlines()
        if not sub._is("EOF"):
            self._error("E1018", f"unexpected text after the expression '{expr_text}' in f-string", tok)
        return expr

    def _parse_sizeof(self, tok: Token) -> object:
        """`sizeof(T)` / `sizeof(expr)`. A lone name could be either, so both are kept."""
        self._expect_op("(")
        offset = 0
        if self._peek(offset).kind == "KEYWORD" and self._peek(offset).lexeme == "const":
            offset += 1
        word = self._peek(offset)
        is_word = word.kind == "IDENT" or (word.kind == "KEYWORD" and word.lexeme in self._CAST_TYPE_WORDS)
        if is_word:
            stars = 0
            probe = offset + 1
            while self._peek(probe).kind == "OP" and self._peek(probe).lexeme == "*":
                stars += 1
                probe += 1
            if self._peek(probe).kind == "OP" and self._peek(probe).lexeme == ")":
                const = "const " if offset else ""
                self.pos += probe
                self._expect_op(")")
                name = const + word.lexeme + "*" * stars
                if stars == 0 and word.kind == "IDENT" and not offset:
                    return ast.SizeofExpr(
                        type_name=word.lexeme,
                        operand=ast.IdentifierExpr(name=word.lexeme, span=self._span(word)),
                        span=self._span(tok),
                    )
                return ast.SizeofExpr(type_name=name, operand=None, span=self._span(tok))
        operand = self._parse_expression()
        self._expect_op(")")
        return ast.SizeofExpr(type_name=None, operand=operand, span=self._span(tok))

    def _parse_lambda_expr(self) -> ast.LambdaExpr:
        """Parse ``fn(params) -> return_type: body`` as an expression."""
        fn_tok = self._prev()
        self._expect_op("(")
        params: list[ast.Param] = []
        if not self._is_op(")"):
            while True:
                params.append(self._parse_param(params))
                if self._match_op(","):
                    continue
                break
        self._expect_op(")")
        return_type = None
        if self._match_op("->"):
            return_type = self._parse_type(stop_ops={":"})
        colon = self._expect_op(":")
        if self._is("NEWLINE"):
            body = self._parse_block()
        else:
            # `fn(x): x * 2` -- a single expression is the return value
            value = self._parse_expression()
            body = [ast.ReturnStmt(value=value, span=self._span_from_expr(value))]
        return ast.LambdaExpr(params=params, return_type=return_type, body=body, span=self._span(fn_tok))

    def _collect_until_newline(self) -> str:
        parts: list[str] = []
        while not self._is("NEWLINE") and not self._is("EOF"):
            parts.append(self._advance().lexeme)
        return " ".join(parts)

    def _consume_newlines(self) -> None:
        while self._match("NEWLINE"):
            pass

    def _consume_required_newline(self, message: str) -> None:
        if self._after_block:
            # A block expression (e.g. lambda) already consumed the trailing
            # newline structure, so no explicit NEWLINE token will be present.
            self._after_block = False
            self._consume_newlines()
            return
        if not self._match("NEWLINE"):
            self._error("E1003", message, self._current())
        self._consume_newlines()

    def _error(self, code: str, message: str, token: Token) -> None:
        line_text = ""
        if 1 <= token.line <= len(self.source_lines):
            line_text = self.source_lines[token.line - 1]
        raise ManvError(diag(code, message, self.file, token.line, token.column, line_text))

    def _span(self, token: Token) -> Span:
        return Span(self.file, token.line, token.column, token.end_line, token.end_column)

    def _span_from_expr(self, expr: object) -> Span:
        expr_span = getattr(expr, "span", None)
        if isinstance(expr_span, Span):
            return expr_span
        tok = self._current()
        return Span(self.file, tok.line, tok.column, tok.end_line, tok.end_column)

    def _extract_leading_docstring(self, statements: list[object]) -> tuple[str | None, list[object]]:
        if not statements:
            return None, statements
        first = statements[0]
        if not isinstance(first, ast.ExprStmt):
            return None, statements
        if not isinstance(first.expr, ast.LiteralExpr) or first.expr.literal_type != "str":
            return None, statements
        docstring = getattr(first.expr, "raw_value", None)
        if docstring is None:
            docstring = first.expr.value
        return str(docstring), statements[1:]

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
        # Any token advance resets the block-end flag unless we're inside
        # _consume_required_newline which checks and clears it explicitly.
        self._after_block = False
        return tok

    def _is(self, kind: str) -> bool:
        return self._current().kind == kind

    def _is_c_declaration_start(self) -> bool:
        """True at `T name` or `T[] name`: a type word (builtin or user-defined) then a name."""
        tok = self._current()
        if not (tok.kind == "IDENT" or (tok.kind == "KEYWORD" and tok.lexeme in C_DECL_TYPES)):
            return False
        offset = 1
        if self._peek(offset).kind == "KEYWORD" and self._peek(offset).lexeme == "const" and self._peek(offset + 1).lexeme == "*":
            offset += 1
        stars = 0
        while self._peek(offset).kind == "OP" and self._peek(offset).lexeme == "*":
            stars += 1
            offset += 1
        while (
            self._peek(offset).kind == "OP"
            and self._peek(offset).lexeme == "["
            and self._peek(offset + 1).kind == "OP"
            and self._peek(offset + 1).lexeme == "]"
        ):
            offset += 2
        if self._peek(offset).kind != "IDENT":
            return False
        if stars:
            # `a * b` alone is a product; only `T* name =`, `T* name[` or `T* name` + newline declares.
            after = self._peek(offset + 1)
            return after.kind in {"NEWLINE", "EOF"} or (after.kind == "OP" and after.lexeme in {"=", "["})
        return True

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
            and self._peek(2).kind in {"IDENT", "KEYWORD"}
            and self._peek(3).kind == "OP"
            and self._peek(3).lexeme == "="
        ):
            return None
        ident = self._advance()
        target = ast.IdentifierExpr(name=ident.lexeme, span=self._span(ident))
        self._expect_op(".")
        attr = self._expect_attr_name()
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

    def _try_parse_aug_attr_assignment(self) -> ast.AugAssignAttrStmt | None:
        aug_ops = {"+=", "-=", "*=", "/=", "%="}
        if not (
            self._is("IDENT")
            and self._peek(1).kind == "OP"
            and self._peek(1).lexeme == "."
            and self._peek(2).kind in {"IDENT", "KEYWORD"}
            and self._peek(3).kind == "OP"
            and self._peek(3).lexeme in aug_ops
        ):
            return None
        ident = self._advance()
        target = ast.IdentifierExpr(name=ident.lexeme, span=self._span(ident))
        self._expect_op(".")
        attr = self._expect_attr_name()
        op_tok = self._advance()
        value = self._parse_expression()
        return ast.AugAssignAttrStmt(target=target, attr=attr.lexeme, op=op_tok.lexeme, value=value, span=self._span(op_tok))

    def _try_parse_aug_index_assignment(self) -> ast.AugAssignIndexStmt | None:
        aug_ops = {"+=", "-=", "*=", "/=", "%="}
        if not (self._is("IDENT") and self._peek(1).kind == "OP" and self._peek(1).lexeme == "["):
            return None
        start = self.pos
        ident_tok = self._advance()
        target = ast.IdentifierExpr(name=ident_tok.lexeme, span=self._span(ident_tok))
        self._expect_op("[")
        index_expr = self._parse_expression()
        self._expect_op("]")
        if not (self._current().kind == "OP" and self._current().lexeme in aug_ops):
            self.pos = start
            return None
        op_tok = self._advance()
        value = self._parse_expression()
        return ast.AugAssignIndexStmt(target=target, index=index_expr, op=op_tok.lexeme, value=value, span=self._span(op_tok))

    def _parse_with_stmt(self) -> ast.WithStmt:
        with_tok = self._prev()
        self._no_cast = True  # `with expr as name`: here `as` binds the name, it is not a cast
        try:
            context = self._parse_expression()
        finally:
            self._no_cast = False
        bind_name: str | None = None
        if self._match_keyword("as"):
            bind_tok = self._expect("IDENT", message="expected bind variable after 'as'")
            bind_name = bind_tok.lexeme
        self._expect_op(":")
        body = self._parse_block()
        return ast.WithStmt(context=context, bind_name=bind_name, body=body, span=self._span(with_tok))

    def _parse_match_stmt(self) -> ast.MatchStmt:
        match_tok = self._prev()
        subject = self._parse_expression()
        self._expect_op(":")
        self._consume_required_newline("expected newline after ':'")
        self._expect("INDENT", message="expected indented match block")
        cases: list[ast.CaseClause] = []
        while not self._is("DEDENT") and not self._is("EOF"):
            self._consume_newlines()
            if self._is("DEDENT") or self._is("EOF"):
                break
            if not (self._is("IDENT") and self._current().lexeme == "case"):
                self._error("E1014", "expected 'case' clause inside match block", self._current())
            case_tok = self._advance()
            pattern = self._parse_range()
            guard: object | None = None
            if self._match_keyword("if"):
                guard = self._parse_expression()
            self._expect_op(":")
            body = self._parse_block()
            cases.append(ast.CaseClause(pattern=pattern, guard=guard, body=body, span=self._span(case_tok)))
        self._expect("DEDENT", message="expected dedent after match block")
        return ast.MatchStmt(subject=subject, cases=cases, span=self._span(match_tok))

    def _expect_attr_name(self) -> Token:
        """An attribute name after `.`: any word, including reserved ones (`e.type`)."""
        if self._is("KEYWORD"):
            return self._advance()
        return self._expect("IDENT", message="expected attribute name")

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
