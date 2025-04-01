import React from "react";

interface ButtonProps {
    label?: string;
    onClick?: () => void;
    className?: string;
    style?: React.CSSProperties;
    icon?: React.ReactNode;
    variant?: "solid" | "outline";
}

const Button: React.FC<ButtonProps> = ({
    label,
    onClick,
    className = "",
    style = {},
    icon,
    variant = "solid",
}) => {
    const baseStyles = `
        px-6 py-3 text-md sm:text-lg capitalize font-medium sm:font-bold rounded-xl transition duration-400 ease-in-out opacity-100 scale-100 hover:scale-95
        inline-flex items-center justify-center cursor-pointer`;

    const solidStyles = "bg-black text-white";
    const outlineStyles = "border border-black text-black bg-transparent p-2";

    const combinedClassName = `${baseStyles} ${
        variant === "outline" ? outlineStyles : solidStyles
    } ${className}`.trim();

    return (
        <button className={combinedClassName} onClick={onClick} style={style}>
            {icon && <span className="scale-140 pr-3">{icon}</span>}
            {label}
        </button>
    );
};

export default Button;
